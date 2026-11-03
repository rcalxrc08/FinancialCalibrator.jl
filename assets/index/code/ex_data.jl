# This file was generated, do not modify it. # hide
id = CalibrationID("Dax", "22Aug2016", "EUR")
und, mkt = adapter(retriever(id, IOFileInputStrategy()))
println("Spot: ", und.S0)
println("Maturities: ", length(mkt.TimeToMaturity), ", strikes: ", length(mkt.Strikes))