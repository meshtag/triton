add_triton_library(TritonIMToLLVM
    TargetInfo.cpp
    TritonIMToLLVM.cpp

    DEPENDS
    TritonIMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUToLLVM
)
