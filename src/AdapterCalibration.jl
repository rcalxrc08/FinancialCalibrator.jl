using JSON,FinancialToolbox;
import Dierckx.Spline1D;
function GeneralAdapterStrategy(SpotString::String,RfString::String,IdString::String,VolaString::String,YfrTreshold::Tuple{Real,Real}=(0.1,5.0),MoneynessTreshold::Tuple{Real,Real}=(0.7,2.0),BasisConvention::Integer=1)
	####Input Parser and Adapter

	# JSON.parse - string or stream to Julia data structures
	jSpot = JSON.parse(SpotString)[1]
	jRf = JSON.parse(RfString)
	jImplDiv = JSON.parse(IdString)
	jVolat = JSON.parse(VolaString)

	######## Parsing of Input Data
	#BasisConvention=1;#ACT/365
	#Spot Parsing
	S0=jSpot["value"]
	RefDate=Date(jSpot["referenceDate"])

	#Zero Rate Parsing
	RfArray=[ jRf[i]["value"]  for i=1:length(jRf) ];
	DatesRfArray=Date.([ jRf[i]["maturityDate"]  for i=1:length(jRf) ]);
	yfrRfArray=[ yearfrac(RefDate,matDate,BasisConvention)  for matDate in DatesRfArray ];
	IndexRf=sortperm(yfrRfArray);
	yfrRfArray=yfrRfArray[IndexRf]
	RfArray=RfArray[IndexRf]

	#Implied Dividend Parsing
	IdArray=[ jImplDiv[i]["value"]  for i=1:length(jImplDiv) ];
	DatesIdArray=Date.([ jImplDiv[i]["maturityDate"]  for i=1:length(jImplDiv) ]);
	yfrIdArray=[ yearfrac(RefDate,matDate,BasisConvention)  for matDate in DatesIdArray ];
	IndexId=sortperm(yfrIdArray);
	yfrIdArray=yfrIdArray[IndexId]
	IdArray=IdArray[IndexId]

	#Implied Vol Parsing
	maturityDates=((sort(unique([ jVolat[i]["maturityDate"] for i in 1:length(jVolat)]))))
	moneyness=sort(unique([ jVolat[i]["pointValue"] for i in 1:length(jVolat)]))
	VolaMatrix=Matrix(undef,length(maturityDates),length(moneyness));

	for i=1:length(maturityDates)
		volaTmp=([jVolat[j]["value"] for j in 1:length(jVolat) if jVolat[j]["maturityDate"]==maturityDates[i]])
		moneyTemp=([jVolat[j]["pointValue"] for j in 1:length(jVolat) if jVolat[j]["maturityDate"]==maturityDates[i]])
		IndexMoney=sortperm(moneyTemp);
		VolaMatrix[i,1:end]=volaTmp[IndexMoney]
	end

	maturityDates=Date.(maturityDates)

	yfrVolaArray=[ yearfrac(RefDate,matDate,BasisConvention)  for matDate in maturityDates ];

	##### Augmenting Phase

	#Augment ZeroRate
	ZeroSplineInterpolator=Dierckx.Spline1D(yfrRfArray,RfArray);
	ZeroRateAug=ZeroSplineInterpolator(yfrVolaArray);

	#Augment ZeroRate
	IdSplineInterpolator=Dierckx.Spline1D(yfrIdArray,IdArray);
	IdAug=IdSplineInterpolator(yfrVolaArray);

	##### Resize All the Inputs
	#Dates
	TimeToMaturity=[yfrVolaArray[i] for i in 1:length(yfrVolaArray) if (yfrVolaArray[i]>YfrTreshold[1])&&(yfrVolaArray[i]<YfrTreshold[2])]
	maturityDates=[maturityDates[i] for i in 1:length(yfrVolaArray) if (yfrVolaArray[i]>YfrTreshold[1])&&(yfrVolaArray[i]<YfrTreshold[2])]

	#Moneyness
	Moneyness=[moneyness[i] for i in 1:length(moneyness) if (moneyness[i]>MoneynessTreshold[1])&&(moneyness[i]<MoneynessTreshold[2])]

	#Zero Rate
	ZeroRates=[ZeroRateAug[i] for i in 1:length(ZeroRateAug) if (yfrVolaArray[i]>YfrTreshold[1])&&(yfrVolaArray[i]<YfrTreshold[2])]
	#Implied Dividend
	ImpliedDividend=[IdAug[i] for i in 1:length(IdAug) if (yfrVolaArray[i]>YfrTreshold[1])&&(yfrVolaArray[i]<YfrTreshold[2])]

	#Volatility Matrix
	Volatility=Matrix{Float64}(undef,length(TimeToMaturity),length(Moneyness));
	for i=1:length(TimeToMaturity)
		volaTmp=[VolaMatrix[i,j] for j in 1:length(moneyness) if (moneyness[j]>MoneynessTreshold[1])&&(moneyness[j]<MoneynessTreshold[2])]
		Volatility[i,1:end]=volaTmp
	end
	Strikes=Moneyness.*S0
	
	return (S0,TimeToMaturity,maturityDates,ZeroRates,ImpliedDividend,Strikes,Volatility)

end


function adapter(jsonMktDataIN::RawMarketData)::MarketData
    ####Input Adapter
	adaptedData=GeneralAdapterStrategy(jsonMktDataIN.SpotString,jsonMktDataIN.RfString,jsonMktDataIN.ImplDivString,jsonMktDataIN.VolatString);
    
    return MarketData(adaptedData[1],adaptedData[2],adaptedData[3],adaptedData[4],adaptedData[5],adaptedData[6],adaptedData[7]);
end

