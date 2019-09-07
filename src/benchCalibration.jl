using BenchmarkTools
include("Calibrator.jl")

####Input Retriever
currency="EUR"
underlying="Dax"
refDate="22Aug2016"

cal=CalibrationID(underlying,refDate,currency);

const rawMarketData=retriever(cal,IOFileInputStrategy());

const mktData=adapter(rawMarketData);

################################################
######## Carr Madan Calibration
## Characteristic Function (Define the Model)
#NIG
#const CharExp(v::Number,p::Array{Float64})::Number=(1-1*sqrt(1.+ ((v.^2)*(p[1]*p[1])-2.*1im*p[2]*v)*p[3]))/p[3];
#Variance Gamma
const CharExp(u::Number,p::Array{Float64})::Number=-1/p[3]*log(1+u*u*p[1]*p[1]*p[3]/2.0-1im*p[2]*p[3]*u);
##Kou Model
#const CharExp(u::Number,p::Array{Float64})::Number=-p[1]*p[1]*u*u/2.0+1im*u*p[2]*(p[3]/(p[4]-1im*u)-(1-p[3])/(p[5]+1im*u));
#Merton Model
#const CharExp(u::Number,p::Array{Float64})::Number=-p[1]*p[1]*u*u/2.0+p[2]*(exp(-p[4]*p[4]*u*u/2.0+1im*p[3]*u)-1);


##Define the Method
calibratorNIG=CalibratorCarrMadan(CharExp);
## Initial Parameters
#Kou starting Point
#InitialPointNIG=Float64[0.3,7.0,0.3,30.0,30.0]
#Merton starting Point
#InitialPointNIG=Float64[0.3,7.0,0.3,30.0,30.0]
#Infinite Activity starting Point
InitialPointNIG=Float64[0.2,-0.33,0.16] 
#Calibration
ModelVolatilityNIG=calibrate(calibratorNIG,mktData,InitialPointNIG)
@btime ModelVolatilityNIG=calibrate(calibratorNIG,mktData,InitialPointNIG)
