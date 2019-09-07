
abstract type InputType end

struct IOFileInputStrategy <: InputType end
struct IOFileInputMDS <: InputType end

struct WebFileInput <: InputType end
struct WebFileInputMDS <: InputType end


struct CalibrationID
    underlying::String
    refDate::String
    currency::String
    CalibrationID(underlyingIn,refDateIn,currencyIn)=new(underlyingIn,refDateIn,currencyIn)
end


struct RawMarketData
	refDate::Date
    SpotString::String
    RfString::String
    ImplDivString::String
    VolatString::String
    RawMarketData(refDate,SpotStringIn,RfStringIn,ImplDivStringIn,VolatStringIn)=new(refDate,SpotStringIn,RfStringIn,ImplDivStringIn,VolatStringIn)
end

struct RawMarketDataMDS
	refDate::Date
    SpotString::String
    RfString::String
    ImplDivString::String
    VolatString::String
    RawMarketDataMDS(refDate,SpotStringIn,RfStringIn,ImplDivStringIn,VolatStringIn)=new(refDate,SpotStringIn,RfStringIn,ImplDivStringIn,VolatStringIn)
end


function IORetriever(SpotRelPath::String,RfRelPath::String,IdRelPath::String,VolaRelPath::String)
	####Input Retriever
	SpotString=read(SpotRelPath,String);
	RfString=read(RfRelPath,String);
	ImplDivString=read(IdRelPath,String);
	VolatString=read(VolaRelPath,String);

	return (SpotString,RfString,ImplDivString,VolatString)
end



function WEBRetriever(SpotRelPath::String,RfRelPath::String,IdRelPath::String,VolaRelPath::String)
	####Input Retriever
	SpotString=readdlm(download(SpotRelPath))[1];
	if(SpotString=="[]")
		error("Data Not Available")
	end
	RfString=readdlm(download(RfRelPath))[1];
	ImplDivString=readdlm(download(IdRelPath))[1];
	VolatString=readdlm(download(VolaRelPath))[1];
	

	return (SpotString,RfString,ImplDivString,VolatString)
end


struct MarketData
    S0::Float64
    TimeToMaturity::Array{Float64}
	MaturityDates::Array{Date}
    ZeroRates::Array{Float64}
    ImpliedDividend::Array{Float64}
    Strikes::Array{Float64}
    Volatility::Matrix{Float64}
    MarketData(S0in,TimeToMaturityin,MaturityDatesIn,ZeroRatesin,ImpliedDividendin,Strikesin,Volatilityin)=new(S0in,TimeToMaturityin,MaturityDatesIn,ZeroRatesin,ImpliedDividendin,Strikesin,Volatilityin)
end