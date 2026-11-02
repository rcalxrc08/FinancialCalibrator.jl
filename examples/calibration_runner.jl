using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT
import FinancialCalibrator: CalibrationID, IOFileInputStrategy, retriever, adapter, calibrate, AnalyticMethod

####Input Retriever
currency="EUR"
underlying="Dax"
refDate="22Aug2016"

cal=CalibrationID(underlying, refDate, currency);

const rawMarketData=retriever(cal, IOFileInputStrategy());

const UnderlyingData, mktData = adapter(rawMarketData);

################################################
######## Carr Madan Calibration

##Define the Method
## Initial Parameters
#Infinite Activity starting Point and Model
modelNIG=NormalInverseGaussianProcess(0.2, -0.03, 0.16, UnderlyingData)
methodNIG=CarrMadanMethod(400.0, 14)
#Calibration

ModelVolatilityNIG=calibrate(modelNIG, methodNIG, mktData)

using PyPlot;
surf_=surf(mktData.TimeToMaturity, mktData.Strikes, ModelVolatilityNIG.ModelVolatility')
surf_=surf(mktData.TimeToMaturity, mktData.Strikes, mktData.Volatility')

surf_._facecolors2d=surf_._facecolors3d
surf_._edgecolors2d=surf_._edgecolors3d

xlabel("TimeToMaturity")
ylabel("Strike")
zlabel("Volatility")
title("Model Calibration "*underlying*" "*refDate)
legend(["Model", "Market"])
