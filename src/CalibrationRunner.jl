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
## Characteristic Function (Define the Model)
#NIG
#const CharExpNIGP(v::Number,p::Array{Float64})::Number=(1-1*sqrt(1.+ ((v.^2)*(p[1]*p[1])-2.*1im*p[2]*v)*p[3]))/p[3];
#Variance Gamma
const CharExpNIGP(u::Number,p::Array{Float64})::Number=-1/p[3]*log(1+u*u*p[1]*p[1]*p[3]/2.0-1im*p[2]*p[3]*u);
##Kou Model
#const CharExpNIGP(u::Number,p::Array{Float64})::Number=-p[1]*p[1]*u*u/2.0+1im*u*p[2]*(p[3]/(p[4]-1im*u)-(1-p[3])/(p[5]+1im*u));
#Merton Model
#const CharExpNIGP(u::Number,p::Array{Float64})::Number=-p[1]*p[1]*u*u/2.0+p[2]*(exp(-p[4]*p[4]*u*u/2.0+1im*p[3]*u)-1);


##Define the Method

## Initial Parameters
#Kou starting Point
#InitialPointNIG=Float64[0.3,7.0,0.3,30.0,30.0]
#Merton starting Point
#InitialPointNIG=Float64[0.3,7.0,0.3,30.0,30.0]
#Infinite Activity starting Point and Model
InitialPointNIG=Float64[0.2,-0.03,0.16] 
calibratorNIG=CalibratorCarrMadan(NormalInverseGaussianProcess(InitialPointNIG[1],InitialPointNIG[2],InitialPointNIG[3]));
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
calibratorShiftedLogNormalMixture=CalibratorShiftedLogNormalMixture();
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