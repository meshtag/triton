/// IMArtifact.cpp. The tables a kernel states to the trace runtimes, as module globals the
/// runtimes read by name at link time. The lowering writes them right before its conversion,
/// which drops the im.residency stamps most of them come from. Facts an earlier pass derives
/// reach here as module attributes, such as im.acc_cells_per_lane and OptiPIM's im.opt_<x>.
///
/// A table is an i32 array global with an i32 count beside it. An empty table is left out and
/// its count says 0.

#include "IMArtifact.h"
#include "IMContraction.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton::im;
using namespace mlir::triton::im::contraction;
namespace ttg = mlir::triton::gpu;

/// An i32 array global. An empty table is left out, and its count says so.
static void emitI32Table(OpBuilder &b, Location loc, StringRef name,
                         ArrayRef<int32_t> values) {
  if (values.empty())
    return;
  Type i32 = b.getI32Type();
  auto dataTy = RankedTensorType::get({(int64_t)values.size()}, i32);
  LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, values.size()),
                         /*isConstant=*/true, LLVM::Linkage::External, name,
                         DenseElementsAttr::get(dataTy, values));
}

/// An i32 scalar global.
static void emitI32(OpBuilder &b, Location loc, StringRef name, int32_t value) {
  LLVM::GlobalOp::create(b, loc, b.getI32Type(), /*isConstant=*/true,
                         LLVM::Linkage::External, name, b.getI32IntegerAttr(value));
}

/// An integer field of an access's im.residency stamp, `dflt` when it is absent.
static int64_t residencyInt(Operation *op, StringRef field, int64_t dflt) {
  auto res = dyn_cast_or_null<DictionaryAttr>(op->getDiscardableAttr("im.residency"));
  auto v = res ? res.getAs<IntegerAttr>(field) : IntegerAttr();
  return v ? v.getInt() : dflt;
}

// --------------------------------------------------------------------------
// The layout table, one record per kernel pointer argument
// --------------------------------------------------------------------------

static constexpr int kMaxAxes = 3;
static constexpr int kAxisWords = 5;
/// A record is { operand_arg, bank_replicated, num_axes, is_store, then kMaxAxes copies of
/// { extent, stride factor, arg, arg, arg }, then live_split and cell_fanout }.
/// bank_replicated is the compiler-derived receive count: 0 when the tensor is partitioned
/// across the banks, otherwise how many banks receive a copy, where 1 (a scalar splat, or a
/// store with no lane axis) reads as every bank. -1 when the pass said nothing.
/// is_store marks the output, which the SIMDRAM occupancy charge needs, since only a store's
/// footprint leaves out the reduction axis.
/// The footprint is logical. A stride the pass could not fold to a number travels as a
/// product of kernel argument indices, and the runtimes read only the extents.
/// live_split is the factor a rank-collapsing reduce hid from the store's footprint, which
/// multiplies the occupancy the SIMDRAM runtime derives. It exceeds 1 where a kernel stores
/// a reduce of a wider carried accumulator, as the reduction-factored matvec and DCC's RED
/// partial kernels do, and emitDccAccGrf and emitDccReturnFrom key on it. live_split and
/// cell_fanout follow the axis padding, so every axis word keeps its index.
/// kRecWords must equal PIM_LAYOUT_REC_WORDS in the runtimes' pim_layout_table.h, which each
/// runtime checks against @__pim_layout_rec_words.
static constexpr int kRecWords = 6 + kMaxAxes * kAxisWords;

/// @__pim_layout_table and @__pim_layout_count from the residency stamps, with the scalars
/// the runtimes check the kernel against.
static void emitPimLayoutTable(ModuleOp mod) {
  struct Entry {
    // Not emitted. The merge rule below needs to know whether a record carries a real
    // classification or the StreamedElementwise catch-all.
    bool concreteClass = false;
    int32_t bankReplicated = -1; // -1 = the pass said nothing
    int32_t isStore = 0;
    int32_t liveSplit = 1;
    int32_t cellFanout = 1;
    SmallVector<int32_t> axes; // kAxisWords per axis, at most kMaxAxes
  };
  llvm::MapVector<int64_t, Entry> byOperand;

  mod.walk([&](Operation *op) {
    auto dict = dyn_cast_or_null<DictionaryAttr>(op->getDiscardableAttr("im.residency"));
    auto cls = dict ? dict.getAs<StringAttr>("reuse_class") : StringAttr();
    int64_t operandArg = residencyInt(op, "operand_arg", -1);
    if (!cls || operandArg < 0)
      return; // no stamp, or an address that resolved to no kernel pointer argument

    Entry e;
    e.isStore = isa<triton::StoreOp>(op) ? 1 : 0;
    e.liveSplit = (int32_t)residencyInt(op, "live_split", 1);
    e.cellFanout = (int32_t)residencyInt(op, "cell_fanout", 1);
    e.concreteClass = cls.getValue() == "BroadcastReplicate" ||
                      cls.getValue() == "ReductionStridedMatrix" ||
                      cls.getValue() == "ParallelSpread";
    e.bankReplicated = (int32_t)residencyInt(op, "bank_replicated", -1);
    if (auto fp = dict.getAs<DenseI64ArrayAttr>("footprint")) {
      ArrayRef<int64_t> v = fp.asArrayRef();
      if (v.size() % kAxisWords == 0 &&
          (int)(v.size() / kAxisWords) <= kMaxAxes)
        for (int64_t x : v)
          e.axes.push_back((int32_t)x);
    }

    // First wins, except a real class beats an earlier catch-all.
    auto it = byOperand.find(operandArg);
    if (it == byOperand.end())
      byOperand.insert({operandArg, e});
    else if (!it->second.concreteClass && e.concreteClass) {
      // A wholesale replace must not drop what the earlier record knew.
      e.isStore |= it->second.isStore;
      if (it->second.liveSplit > e.liveSplit)
        e.liveSplit = it->second.liveSplit;
      if (it->second.cellFanout > e.cellFanout)
        e.cellFanout = it->second.cellFanout;
      it->second = e;
    }
    else if (it->second.bankReplicated < 0 && e.bankReplicated >= 0)
      it->second.bankReplicated = e.bankReplicated;
    // Sticky, unlike the rest: an RMW pointer arg is loaded AND stored, and
    // first-wins would drop the store bit whichever order the walk saw them in.
    if (e.isStore)
      byOperand[operandArg].isStore = 1;
    // MAX, not first-wins: two stores can reach one output (CO-fused conv) and the
    // row has to hold the widest of them. Never sum, since they share the accumulator.
    if (e.liveSplit > byOperand[operandArg].liveSplit)
      byOperand[operandArg].liveSplit = e.liveSplit;
    if (e.cellFanout > byOperand[operandArg].cellFanout)
      byOperand[operandArg].cellFanout = e.cellFanout;
  });

  if (byOperand.empty())
    return;

  SmallVector<int32_t> flat;
  flat.reserve(byOperand.size() * kRecWords);
  for (const auto &kv : byOperand) {
    flat.push_back((int32_t)kv.first);
    flat.push_back(kv.second.bankReplicated);
    flat.push_back((int32_t)(kv.second.axes.size() / kAxisWords));
    flat.push_back(kv.second.isStore);
    flat.append(kv.second.axes.begin(), kv.second.axes.end());
    flat.resize(flat.size() + (kMaxAxes * kAxisWords - kv.second.axes.size()),
                0);
    flat.push_back(kv.second.liveSplit);
    flat.push_back(kv.second.cellFanout);
  }

  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  emitI32Table(b, loc, "__pim_layout_table", flat);
  emitI32(b, loc, "__pim_layout_count", (int32_t)byOperand.size());
  // The record width, which the runtime checks against PIM_LAYOUT_REC_WORDS. The two live in
  // separate submodules, and a runtime striding at another width misreads every record past
  // the first.
  emitI32(b, loc, "__pim_layout_rec_words", kRecWords);
  // The bus width the vector widths were chosen for, which the runtime checks against its
  // cfg_dq_bits.
  emitI32(b, loc, "__pim_dq_bits", (int32_t)moduleDqBits(mod));

  auto moduleInt = [&](StringRef attr) -> int32_t {
    auto a = mod->getAttrOfType<IntegerAttr>(attr);
    return a ? (int32_t)a.getInt() : 0;
  };
  // Placement policy, 0 = the kernel said nothing. The dcc-parity runtime refuses anything
  // but row-pack, and the default and optipim-parity runtimes refuse anything but
  // interleaved at DQ alignment.
  emitI32(b, loc, "__pim_persistent", moduleInt("im.persistent"));
  emitI32(b, loc, "__pim_layout_scheme", moduleInt("im.layout_scheme"));
  emitI32(b, loc, "__pim_placement_align", moduleInt("im.placement_align"));
  // Cells one lane keeps live across a reduction, the widest loop-carried accumulator.
  // 0 when no store drains one, which is every elementwise kernel.
  emitI32(b, loc, "__pim_acc_cells_per_lane", moduleInt("im.acc_cells_per_lane"));
  // Entries per lane of the widest strip im-load-cluster jammed in a storing loop, 0 none.
  emitI32(b, loc, "__pim_map_strip_entries", moduleInt("im.map_strip_entries"));
  // The lane count the kernel was compiled for, which the runtimes check against their own
  // bank count and the lanes the harness replays.
  emitI32(b, loc, "__pim_lanes",
          (int32_t)ttg::TritonGPUDialect::getThreadsPerWarp(mod));
}

// --------------------------------------------------------------------------
// Anaheim's placement, for the anaheim runtime
// --------------------------------------------------------------------------

/// The PolyGroup of each pointer argument, as im_polygroups states it, so the anaheim
/// runtime places by the compiler's decision, and the polynomials per PolyGroup of a split
/// tensor (im.polygroup_split, 0 unsplit). Both tables are emitted whenever the option is
/// set: a library that left one to the runtime's weak default could have dyld bind it to
/// another kernel's table loaded in the same process. Absent unless the option is set.
static void emitAnaheimPolyGroups(ModuleOp mod) {
  auto groups = mod->getAttrOfType<DenseI64ArrayAttr>("im.polygroups");
  if (!groups)
    return;
  SmallVector<int32_t> flat;
  for (int64_t g : groups.asArrayRef())
    flat.push_back((int32_t)g);
  SmallVector<int32_t> splitFlat(flat.size(), 0);
  if (auto split = mod->getAttrOfType<DenseI64ArrayAttr>("im.polygroup_split"))
    for (auto [i, n] : llvm::enumerate(split.asArrayRef()))
      if (i < splitFlat.size())
        splitFlat[i] = (int32_t)n;
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  emitI32Table(b, loc, "__anaheim_polygroups", flat);
  emitI32(b, loc, "__anaheim_polygroup_count", (int32_t)flat.size());
  emitI32Table(b, loc, "__anaheim_polygroup_split", splitFlat);
  emitI32(b, loc, "__anaheim_polygroup_split_count", (int32_t)splitFlat.size());
}

// --------------------------------------------------------------------------
// DCC's conventions, for the dcc-parity runtime
// --------------------------------------------------------------------------

// DCC's tile MAC per tensor, [operand_arg, reads per MAC] pairs, only when the module
// asked for it (im-dcc-tile-mac). Absent otherwise, so the default artifact is unchanged.
static void emitDccTileMac(ModuleOp mod) {
  if (!mod->hasAttr("im.dcc_tile_mac"))
    return;
  llvm::MapVector<int64_t, int32_t> byArg;
  mod.walk([&](triton::LoadOp ld) {
    auto stamp = ld->getAttrOfType<DenseI64ArrayAttr>("im.dcc-tile-mac");
    int64_t arg = residencyInt(ld, "operand_arg", -1);
    if (stamp && arg >= 0)
      byArg[arg] = (int32_t)stamp.asArrayRef()[0];
  });
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  SmallVector<int32_t> flat;
  for (const auto &kv : byArg)
    flat.append({(int32_t)kv.first, kv.second});
  emitI32Table(b, loc, "__pim_dcc_tile_mac", flat);
  emitI32(b, loc, "__pim_dcc_tile_mac_count", (int32_t)byArg.size());
  // DCC's MAC addressing (im.dcc_mac_addressing): [operand_arg, head stride, reduce
  // stride, reduce extent] per tensor, the geometry their generator addresses a tile by.
  if (!mod->hasAttr("im.dcc_mac_addressing"))
    return;
  llvm::MapVector<int64_t, SmallVector<int32_t>> geo;
  mod.walk([&](triton::LoadOp ld) {
    auto g = ld->getAttrOfType<DenseI64ArrayAttr>("im.dcc-mac-addr");
    int64_t arg = residencyInt(ld, "operand_arg", -1);
    if (g && arg >= 0)
      geo[arg] = {(int32_t)g.asArrayRef()[0], (int32_t)g.asArrayRef()[1],
                  (int32_t)g.asArrayRef()[2]};
  });
  SmallVector<int32_t> geoFlat;
  for (const auto &kv : geo) {
    geoFlat.push_back((int32_t)kv.first);
    geoFlat.append(kv.second.begin(), kv.second.end());
  }
  emitI32Table(b, loc, "__pim_dcc_mac_addr", geoFlat);
  emitI32(b, loc, "__pim_dcc_mac_addr_count", (int32_t)geo.size());
}

// DCC's accumulator convention per stored tensor and its GRF_A operand election, derived
// from the residency stamps, only when the module asked for it (im.dcc_acc_grf). A map, or
// a store folded across a column's SIMD lanes, is DCC's result-register path: no reset,
// one write-back per PCU pair. Any other store is an accumulator, reset and written back
// in every lane. GRF_A holds a partitioned input whose loaded values each feed several
// cells of the lane, when another partitioned input feeds one.
static LogicalResult emitDccAccGrf(ModuleOp mod) {
  if (!mod->hasAttr("im.dcc_acc_grf"))
    return success();
  auto accAttr = mod->getAttrOfType<IntegerAttr>("im.acc_cells_per_lane");
  int64_t accCells = accAttr ? accAttr.getInt() : 0;
  // A store of a reduce im-lane-fold marked folds a column's lanes too. A slice shorter than
  // its loop step leaves no loop, so live_split, which reads a carried accumulator, misses it.
  auto foldsColumn = [](Value v) {
    for (int hops = 0; v && hops < 8; ++hops) {
      Operation *def = v.getDefiningOp();
      if (!def)
        return false;
      if (isa<triton::ReduceOp>(def))
        return def->hasAttr("im.lane-fold");
      if (def->getNumOperands() != 1)
        return false;
      v = def->getOperand(0);
    }
    return false;
  };
  // A map's value is elementwise arithmetic on this iteration's loads: no reduction and no
  // value carried across a loop. Its store is the result path even beside a reduction, as
  // the z = y + b a fused VA -> RED writes before it reads z back.
  auto isMap = [](Value v) {
    SmallVector<Value> work{v};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
      Value x = work.pop_back_val();
      if (!seen.insert(x).second)
        continue;
      Operation *def = x.getDefiningOp();
      if (!def)
        return false;
      if (isa<triton::LoadOp>(def))
        continue;
      if (isa<triton::ReduceOp>(def))
        return false;
      for (Value o : def->getOperands())
        work.push_back(o);
    }
    return true;
  };
  llvm::MapVector<int64_t, std::pair<int32_t, int32_t>> acc;
  // A map's store beside a reduction, which the runtime cannot tell from the module.
  llvm::SetVector<int64_t> maps;
  bool conflict = false;
  mod.walk([&](triton::StoreOp st) {
    int64_t arg = residencyInt(st, "operand_arg", -1);
    if (arg < 0)
      return;
    bool map = accCells > 0 && isMap(st.getValue());
    if (map)
      maps.insert(arg);
    bool result = accCells == 0 || residencyInt(st, "live_split", 1) > 1 ||
                  foldsColumn(st.getValue()) || map;
    std::pair<int32_t, int32_t> c = result ? std::make_pair(0, 2) : std::make_pair(1, 1);
    auto it = acc.find(arg);
    if (it != acc.end() && it->second != c)
      conflict = true;
    acc[arg] = c;
  });
  // Per argument: every load partitioned, and the most cells one value feeds.
  llvm::MapVector<int64_t, std::pair<bool, int64_t>> loads;
  mod.walk([&](triton::LoadOp ld) {
    int64_t arg = residencyInt(ld, "operand_arg", -1);
    if (arg < 0)
      return;
    auto &e = loads.insert({arg, {true, INT64_MAX}}).first->second;
    e.first &= residencyInt(ld, "bank_replicated", -1) == 0;
    e.second = std::min(e.second, residencyInt(ld, "cell_fanout", 1));
  });
  bool streamed = llvm::any_of(loads, [](auto &kv) {
    return kv.second.first && kv.second.second == 1;
  });
  SmallVector<int32_t> grf;
  for (auto &kv : loads)
    if (streamed && kv.second.first && kv.second.second > 1)
      grf.push_back((int32_t)kv.first);
  if (conflict)
    return mod.emitError() << "im.dcc_acc_grf: one argument is stored both as a map result "
                              "and as an accumulator, which no single DCC convention covers";
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  SmallVector<int32_t> flat;
  for (auto &kv : acc)
    flat.append({(int32_t)kv.first, kv.second.first, kv.second.second});
  emitI32Table(b, loc, "__pim_dcc_acc", flat);
  emitI32(b, loc, "__pim_dcc_acc_count", (int32_t)acc.size());
  emitI32Table(b, loc, "__pim_dcc_grf_a", grf);
  emitI32(b, loc, "__pim_dcc_grf_a_count", (int32_t)grf.size());
  SmallVector<int32_t> flatMaps(maps.begin(), maps.end());
  emitI32Table(b, loc, "__pim_dcc_map", flatMaps);
  emitI32(b, loc, "__pim_dcc_map_count", (int32_t)flatMaps.size());
  emitI32(b, loc, "__pim_dcc_decided", 1);
  return success();
}

// The arguments whose loads feed `v`. This runs after scf-to-cf, so a loop's carried
// value is a block argument fed by its predecessors' branch operands.
static void loadArgsFeeding(Value v, llvm::SetVector<int64_t> &args) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value x = work.pop_back_val();
    if (!seen.insert(x).second)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(x)) {
      Block *blk = ba.getOwner();
      for (Block *pred : blk->getPredecessors())
        if (auto br = dyn_cast<BranchOpInterface>(pred->getTerminator()))
          for (unsigned s = 0; s < br->getNumSuccessors(); ++s)
            if (br->getSuccessor(s) == blk)
              if (Value in = br.getSuccessorOperands(s)[ba.getArgNumber()])
                work.push_back(in);
      continue;
    }
    Operation *def = x.getDefiningOp();
    if (isa<triton::LoadOp>(def)) {
      if (int64_t a = residencyInt(def, "operand_arg", -1); a >= 0)
        args.insert(a);
      continue;
    }
    for (Value o : def->getOperands())
      work.push_back(o);
  }
}

// DCC's return stage sized from the input (im.dcc_return_from_input): a store of partials
// folded across a column's lanes, whose value is fed by the loads of one other argument,
// returns as many columns per bank as that input spans, as their RED does. [store_arg,
// source_arg] pairs. Partials fed by two arguments, VA -> RED summing y + b, are not stated,
// and a reduction of z read back after the kernel stored it is.
static void emitDccReturnFrom(ModuleOp mod) {
  if (!mod->hasAttr("im.dcc_return_from_input"))
    return;
  SmallVector<int32_t> flat;
  mod.walk([&](triton::StoreOp st) {
    int64_t arg = residencyInt(st, "operand_arg", -1);
    if (arg < 0 || residencyInt(st, "live_split", 1) <= 1)
      return;
    llvm::SetVector<int64_t> fed;
    loadArgsFeeding(st.getValue(), fed);
    fed.remove(arg);
    if (fed.size() != 1)
      return;
    flat.push_back((int32_t)arg);
    flat.push_back((int32_t)fed.front());
  });
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  emitI32Table(b, loc, "__pim_dcc_return_from", flat);
  emitI32(b, loc, "__pim_dcc_return_from_count", (int32_t)(flat.size() / 2));
}

// --------------------------------------------------------------------------
// OptiPIM's conventions, for the optipim-parity runtime
// --------------------------------------------------------------------------

// OptiPIM's codegen_new conventions for the kernel's one contraction (im.optipim_conv):
// the role their tensor index fixes for each operand, Inputs1 written over the bus,
// Inputs2/Filters bank-read, Outputs read back, and the product's elements per lane,
// which only the harness records, as compiler_cells. Their roles are positional, so ours
// are too: the product's first factor is the input and its second the weight.
static LogicalResult emitOptiPimConv(ModuleOp mod) {
  if (!mod->hasAttr("im.optipim_conv"))
    return success();
  FailureOr<Contraction> found = findOneContraction(mod, "im.optipim_conv needs");
  if (failed(found))
    return failure();
  triton::LoadOp in = found->input, wt = found->weight;
  // The argument the residency pass named, else the walk through a tt.reshape it stops at.
  auto argOf = [](Operation *op, Value ptr) -> int64_t {
    if (int64_t a = residencyInt(op, "operand_arg", -1); a >= 0)
      return a;
    return kernelArgOf(ptr);
  };
  int64_t inArg = in ? argOf(in, in.getPtr()) : -1;
  int64_t wtArg = wt ? argOf(wt, wt.getPtr()) : -1;
  triton::StoreOp out = storeOf(found->reduce);
  int64_t outArg = out ? argOf(out, out.getPtr()) : -1;
  if (inArg < 0 || wtArg < 0 || outArg < 0 || inArg == wtArg)
    return mod.emitError() << "im.optipim_conv cannot name the contraction's operands: "
                           << "input arg " << inArg << ", weight arg " << wtArg
                           << ", output arg " << outArg;
  // Iteration points per lane: the product's elements in one lane, less those a padding
  // mask switches off. A conjunct `make_range < constant` on a dimension the lane holds
  // whole bounds that dimension, and a mask on a lane dimension only idles whole lanes.
  auto prodTy = cast<RankedTensorType>(found->product.getType());
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(prodTy.getEncoding());
  if (!blocked)
    return mod.emitError() << "im.optipim_conv needs a blocked product layout";
  SmallVector<int64_t> active(blocked.getSizePerThread().begin(),
                              blocked.getSizePerThread().end());
  auto tpw = blocked.getThreadsPerWarp();
  LogicalResult counted = success();
  auto bound = [&](triton::LoadOp ld) {
    Value mask = ld.getMask();
    auto ldTy = dyn_cast<RankedTensorType>(ld.getType());
    if (!mask || !ldTy)
      return;
    if (ldTy.getRank() != prodTy.getRank()) {
      counted = ld.emitError() << "im.optipim_conv counts a masked operand only when it is "
                                  "loaded at the product's rank";
      return;
    }
    for (auto t : maskTerms(mask, ldTy.getRank())) {
      if (t.opaque || t.constant)
        continue; // a constant conjunct, or a block argument it cannot see into
      if (t.dim >= 0 && tpw[t.dim] != 1)
        continue; // on a lane axis: idles whole lanes, counts nothing in one
      if (t.dim < 0 || !t.pure || !t.bound ||
          t.cmp.getPredicate() != arith::CmpIPredicate::slt) {
        counted = ld.emitError() << "im.optipim_conv cannot count a mask term on a "
                                    "dimension the lane holds whole; write it as "
                                    "`range < extent` on that dimension";
        return;
      }
      active[t.dim] = std::min<int64_t>(active[t.dim], std::max<int64_t>(*t.bound, 0));
    }
  };
  bound(in);
  bound(wt);
  if (failed(counted))
    return failure();
  int64_t cells64 = 1;
  for (int64_t v : active)
    cells64 *= v;
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  emitI32Table(b, loc, "__pim_opt_role",
               {(int32_t)inArg, 1, (int32_t)wtArg, 2, (int32_t)outArg, 3});
  emitI32(b, loc, "__pim_opt_role_count", 3);
  emitI32(b, loc, "__pim_opt_cells", (int32_t)cells64);
  emitI32(b, loc, "__pim_opt_decided", 1);
  return success();
}

// OptiPIM's statements that im-operand-residency-layout and im-optipim-mac stamp on the
// module as im.opt_<x>, each stated as @__pim_opt_<x>.
static void emitOptiPimStatements(ModuleOp mod) {
  OpBuilder b = OpBuilder::atBlockBegin(mod.getBody());
  Location loc = mod.getLoc();
  auto stamp = [&](StringRef name) { return mod->getAttrOfType<DenseI64ArrayAttr>(name); };
  auto table = [&](StringRef name, DenseI64ArrayAttr v) {
    emitI32Table(b, loc, name, SmallVector<int32_t>(v.asArrayRef()));
  };
  // A table of pairs and its pair count.
  auto pairs = [&](StringRef name, DenseI64ArrayAttr v) {
    table(name, v);
    emitI32(b, loc, (name + "_count").str(), (int32_t)(v.size() / 2));
  };
  if (auto macs = stamp("im.opt_macs")) {
    table("__pim_opt_macs", macs);
    emitI32(b, loc, "__pim_opt_macs_count", (int32_t)macs.size());
  }
  if (auto geom = stamp("im.opt_mac_geom"))
    table("__pim_opt_mac_geom", geom);
  if (auto vertical = mod->getAttrOfType<IntegerAttr>("im.opt_vertical"))
    emitI32(b, loc, "__pim_opt_vertical", (int32_t)vertical.getInt());
  if (auto step = stamp("im.opt_step"))
    table("__pim_opt_step", step);
  if (auto hold = stamp("im.opt_hold")) {
    table("__pim_opt_hold", hold);
    emitI32(b, loc, "__pim_opt_hold_count", 1);
    // Their memo's hash of a step as (bound, weight) pairs, slowest loop first.
    if (auto memo = stamp("im.optipim_memo"))
      pairs("__pim_opt_memo", memo);
  }
  if (auto residency = stamp("im.opt_residency"))
    pairs("__pim_opt_residency", residency);
  if (auto rep = stamp("im.opt_lane_rep"))
    pairs("__pim_opt_lane_rep", rep);
  if (auto ext = stamp("im.opt_lane_ext"))
    pairs("__pim_opt_lane_ext", ext);
  if (auto on = stamp("im.opt_lane_on"))
    pairs("__pim_opt_lane_on", on);
}

// --------------------------------------------------------------------------
// Ops no bank can run
// --------------------------------------------------------------------------

// Banks are lanes, and a bank cannot read another bank's registers, so the target's shuffles
// return their input (TargetInfo.cpp). Anything that would move a value between lanes then
// lowers to a wrong value with no error: a reduce or scan with banks on its axis combines
// each lane with itself once per lane bit, and a conversion that moves data across lanes keeps
// the lane's own value. A partitioned tensor whose layout copies it across lanes is two banks
// reading or writing one element. Each is refused here, on the layouts the lowering will use.
static LogicalResult refuseCrossBank(ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  auto lane = StringAttr::get(ctx, "lane"), warp = StringAttr::get(ctx, "warp");
  bool bad = false;
  // How many banks the placement spreads `v`'s `axis` over.
  auto banksAlong = [&](Value v, unsigned axis) -> int64_t {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || !ty.getEncoding())
      return 1;
    triton::LinearLayout ll = ttg::toLinearLayout(ty);
    int64_t n = 1;
    for (StringAttr dim : {lane, warp})
      if (ll.hasInDim(dim))
        for (const auto &basis : ll.getBases().lookup(dim))
          if (basis[axis] != 0)
            n *= 2;
    return n;
  };
  mod.walk([&](Operation *op) {
    if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
      auto red = dyn_cast<triton::ReduceOp>(op);
      unsigned axis = red ? red.getAxis() : cast<triton::ScanOp>(op).getAxis();
      if (int64_t n = banksAlong(op->getOperand(0), axis); n > 1) {
        op->emitError() << "this " << (red ? "reduce" : "scan") << " runs over axis " << axis
                        << ", which the bank placement spreads over " << n << " banks, and "
                        << "a bank cannot read another bank's data. "
                        << (red ? "Reduce within a bank and combine the partials in a "
                                  "second kernel."
                                : "Scan within a bank and carry each bank's total over in a "
                                  "second kernel.");
        bad = true;
      }
    } else if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
      auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
      if (!srcTy || !dstTy || !srcTy.getEncoding() || !dstTy.getEncoding())
        return;
      triton::LinearLayout c = minimalCvtLayout(srcTy, dstTy);
      if (c.hasInDim(lane) || c.hasInDim(warp)) {
        cvt.emitError() << "this layout conversion moves values between banks, which no "
                        << "bank can do. The lane placement upstream of it is not one the "
                        << "target can run.";
        bad = true;
      }
    } else if (isa<triton::LoadOp, triton::StoreOp>(op)) {
      Value v = isa<triton::LoadOp>(op) ? op->getResult(0) : op->getOperand(1);
      auto ty = dyn_cast<RankedTensorType>(v.getType());
      if (residencyInt(op, "bank_replicated", -1) != 0 || !ty || !ty.getEncoding())
        return;
      triton::LinearLayout ll = ttg::toLinearLayout(ty);
      if (!ll.hasInDim(lane))
        return;
      for (const auto &basis : ll.getBases().lookup(lane))
        if (llvm::all_of(basis, [](int32_t x) { return x == 0; })) {
          op->emitError() << "this tensor is partitioned across banks, but its layout puts "
                          << "the same elements in two banks.";
          bad = true;
          return;
        }
    }
  });
  return failure(bad);
}

LogicalResult mlir::triton::im::emitArtifact(ModuleOp mod) {
  if (failed(refuseCrossBank(mod)))
    return failure();
  // Each emitter puts its globals at the top of the module, ahead of the earlier ones, so the
  // LLVM IR lists them in the reverse of this order.
  emitOptiPimStatements(mod);
  emitPimLayoutTable(mod);
  emitAnaheimPolyGroups(mod);
  emitDccTileMac(mod);
  if (failed(emitDccAccGrf(mod)))
    return failure();
  emitDccReturnFrom(mod);
  return emitOptiPimConv(mod);
}
