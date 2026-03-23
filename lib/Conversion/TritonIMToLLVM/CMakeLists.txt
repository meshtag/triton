add_triton_library(TritonIMToLLVM
    RewriteIMLayout.cpp
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
    TritonGPUTransforms
)
