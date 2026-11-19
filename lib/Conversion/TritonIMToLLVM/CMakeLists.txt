add_triton_library(TritonIMToLLVM
    IMArtifact.cpp
    IMOperandResidencyLayout.cpp
    IMLaneFold.cpp
    IMLoadCluster.cpp
    IMDccTileMac.cpp
    IMOptiPimMac.cpp
    IMReluOpcode.cpp
    IMTileBoundary.cpp
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
