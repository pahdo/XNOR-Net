-- pahdonet_a.lua binarizes only the later half of layers, which contains 88% of the parameters

function createModel()
   require 'cudnn'
      local function activation()
      local C= nn.Sequential()
      C:add(nn.BinActiveZ())
      return C
   end

   local function MaxPooling(kW, kH, dW, dH, padW, padH)
    return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end

   local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
         local C= nn.Sequential()
          C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          C:add(activation())
   		  C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
   		 return C
   end

    local function BinMaxConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,mW,mH)
         local C= nn.Sequential()
          C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          C:add(activation())
          C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
          C:add(MaxPooling(3,3,2,2))
       return C
   end
   
local features = nn.Sequential()
   
   -- Spatial = 2D Convolution
   -- Layer 1: SpatialConv->SpatialBatchNorm->ReLU->MaxPooling
   -- SpatialConvolution(inChannels, outChannels, kernelWidth, kernelHeight, stepSizeW, stepSizeH, AdditionalZerosAddedToTheInputPlaneOnBothSidesOfTheWidthAxis, AdditionalZerosAddingToTheInputPlaneOnBothSidesOfTheHeightAxis)
   features:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2)) -- 224 -> 55
   -- has 3 * 11 * 11 * 96 = 34848 parameters // Not a binary conv
   features:add(nn.SpatialBatchNormalization(96,1e-5,false)) 
   features:add(cudnn.ReLU(true))
   features:add(MaxPooling(3,3,2,2)) -- 55 -> 27

   -- Layer 2: SpatialBatchNorm->BinActivZ->SpatialConv->MaxPooling
   features:add(BinMaxConvolution(96,256,5,5,1,1,2,2)) -- 27 -> 27 / 27 -> 13
   -- has 96 * 5 * 5 * 256 = 614400 parameters

   -- Layer 3: SpatialBatchNorm->BinActivZ->SpatialConv
   features:add(BinConvolution(256,384,3,3,1,1,1,1)) -- 13 -> 13
   -- has 256 * 3 * 3 * 384 = 884736 parameters

   -- Layer 4: SpatialBatchNorm->BinActivZ->SpatialConv
   features:add(BinConvolution(384,384,3,3,1,1,1,1)) -- 13 -> 13 (start of my own analysis)
   -- has 384 * 3 * 3 * 384 = 1.327104m parameters

   -- Layer 5: SpatialBatchNorm->BinActivZ->SpatialConv->MaxPooling
   features:add(BinMaxConvolution(384,256,3,3,1,1,1,1)) -- 13 -> 13 
   -- has 384 * 3 * 3 * 256 = 884736 parameters

   -- Layer 6: SpatialBatchNorm->BinActivZ->SpatialConv
   features:add(BinConvolution(256,4096,6,6)) -- 13 -> 8 
   -- has 256 * 6 * 6 * 4096 = 37.748736m parameters

   -- Layer 7: SpatialBatchNorm->BinActivZ=>SpatialConv(1,1)
   features:add(BinConvolution(4096,4096,1,1)) -- 8 -> 8
   -- has 4096 x 4096 = 16.777216m parameters

   -- Layer 8: SpatialBatchNorm->ReLU->SpatialConv(1,1) // Not a binary conv
   features:add(nn.SpatialBatchNormalization(4096,1e-3,false))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(4096,nClasses,1,1)) -- 8 -> 8
   -- nClasses = 1000
   -- has 4096 * 1000 = 4.096m parameters

   -- Layer 9: Reshape->Softmax
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax()) -- Softmax layer
   -- Total = 62.367776m parameters vs. XNOR-Net paper states 61m parameters
 
   local model = features
   return model
end