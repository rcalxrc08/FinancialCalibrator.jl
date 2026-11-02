using FinancialCalibrator, FinancialMonteCarlo
import FinancialCalibrator: CalibrationID, IOFileInputStrategy, retriever, adapter, calibrate, CalibratorCarrMadan, CalibratorShiftedLogNormalMixture

####Input Retriever
currency="EUR"
underlying="Dax"
refDate="22Aug2016"

cal=CalibrationID(underlying, refDate, currency);

#const rawMarketData=retriever(cal,IOFileInputMDS());
const rawMarketData=retriever(cal, IOFileInputStrategy());

const mktData=adapter(rawMarketData);

################################################
######## Carr Madan Calibration

##Define the Method
dividend_curve=FinancialMonteCarlo.ImpliedCurve(mktData.ImpliedDividend, mktData.TimeToMaturity);
#for (t,d) in zip(mktData.TimeToMaturity,mktData.ImpliedDividend)
#	insert!(dividend_curve,t,d)
#end
UnderlyingData=FinancialMonteCarlo.Underlying(mktData.S0, dividend_curve);
## Initial Parameters
#Infinite Activity starting Point and Model
calibratorNIG=CalibratorCarrMadan(NormalInverseGaussianProcess(0.2, -0.03, 0.16, UnderlyingData));
#Calibration

ModelVolatilityNIG=calibrate(calibratorNIG, mktData)

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
