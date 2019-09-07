include("structs.jl")

function retriever(calId::CalibrationID,ioInput::IOFileInputStrategy)::RawMarketData
    ####Input Retriever
    currency=calId.currency
    underlying=calId.underlying
    refDate=calId.refDate
    path=joinpath(pwd(),"json");
    SpotRelPath=joinpath(path,"Spot"*underlying*refDate*".json");
    RfRelPath=joinpath(path,"Rf"*currency*refDate*".json");
    ImplDivRelPath=joinpath(path,"ImplDiv"*underlying*refDate*".json");
    VolaRelPath= joinpath(path,"Volatility"*underlying*refDate*".json");
	TupleString=IORetriever(SpotRelPath,RfRelPath,ImplDivRelPath,VolaRelPath);
	
	adjRefDate=refDate[1:2]*"-"*refDate[3:5]*"-"*refDate[6:end];
	
	MyDateFormat=DateFormat("d-u-y");
	RefDate=Date(adjRefDate,MyDateFormat);
	MyDateFormat=DateFormat("d-u-y");
	RefDate=Date(adjRefDate,MyDateFormat);
    rawMarketData=RawMarketData(RefDate,TupleString[1],TupleString[2],TupleString[3],TupleString[4]);
    return rawMarketData;
end