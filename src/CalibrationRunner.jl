include("Calibrator.jl")

####Input Retriever
currency="EUR"
underlying="Dax"
refDate="22Aug2016"

cal=CalibrationID(underlying,refDate,currency);

#const rawMarketData=retriever(cal,IOFileInputMDS());
const rawMarketData=retriever(cal,IOFileInputStrategy());

const mktData=adapter(rawMarketData);

################################################
######## Carr Madan Calibration


##Define the Method

## Initial Parameters
#Infinite Activity starting Point and Model
calibratorNIG=CalibratorCarrMadan(NormalInverseGaussianProcess(0.2,-0.03,0.16));
#Calibration
ModelVolatilityNIG=calibrate(calibratorNIG,mktData)

################################################
########Shifted LogNormal Mixture Model
## Initial Parameters
Neta=4;
Random.seed!(0)
eta0=0.2*ones(Neta);
lambda0=rand()*ones(Neta-1)/Neta;
alpha0=-0.22;

InitialPointSLNM=Array{Float64}(undef,length(eta0)+length(lambda0)+1)
InitialPointSLNM[1:length(eta0)]=eta0;
InitialPointSLNM[length(eta0)+1:end-1]=lambda0;
InitialPointSLNM[end]=alpha0;
## Define the Model
calibratorShiftedLogNormalMixture=CalibratorShiftedLogNormalMixture(LogNormalMixture);
#Calibration
ModelVolatilitySLNM=calibrate(calibratorShiftedLogNormalMixture,mktData,InitialPointSLNM)



####Plotting



using PyPlot;
surf(mktData.TimeToMaturity,mktData.Strikes,ModelVolatilityNIG.ModelVolatility')
surf(mktData.TimeToMaturity,mktData.Strikes,mktData.Volatility')

xlabel("TimeToMaturity")
ylabel("Strike")
zlabel("Volatility")
title("Model Calibration "*underlying*" "*refDate)
legend(["Model","Market"])
legend(("Model","Market"))