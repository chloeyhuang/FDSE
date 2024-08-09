using ProfileView, BenchmarkTools

using OceanBioME, Oceananigans
import OceanBioME: BoxModelGrid
using OceanBioME: NPZDModel, LOBSTERModel
using Oceananigans.Units
using Oceananigans.Fields: FunctionField

using Plots

include("EKPUtils_old.jl")
include("PZ.jl")
#include("PISCES/PISCES.jl")

println("---------")

function test(model)
    RunBoxModel(model; Δt = 20minutes, stop_time = 20000minutes)
    println(model.clock)
    model.clock.iteration = 0
    model.clock.time = 0
    return model.clock
end

const year = years = 365day

#config for models
PAR⁰(t) = 60 * (1 - cos((t + 15days) * 2π / year)) * (1 / (1 + 0.2 * exp(-((mod(t, year) - 200days) / 50days)^2))) + 2
z = -10 # specify the nominal depth of the box for the PAR profile
PAR_func(t) = PAR⁰(t) * exp(0.2z) # Modify the PAR based on the nominal depth and exponential decay

clock = Clock(time = Float64(0))
grid = BoxModelGrid
PAR = FunctionField{Center, Center, Center}(PAR_func, grid; clock)


bgc = LOBSTER(; grid = BoxModelGrid, light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
model = BoxModel(; biogeochemistry = bgc, clock)
set!(model, NO₃ = 10.0, NH₄ = 0.1, P = 0.1, Z = 0.01)

display(keys(get_params(bgc)))

data = RunBoxModel(model; Δt = 5minutes, stop_time = 50000minutes)
times = data[1]
timeseries = data[2]

plot_timeseries(times, timeseries, timeseries)

