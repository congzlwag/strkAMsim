function get_Paxis(Np::Int; dp::Real=-1, Pmax::Real=-1)::Vector{Float64}
    N_p = Np/2;
    if dp < 0
        @assert Pmax>0 "One of dp or Pmax must be positive"
        dp = 2*Pmax/Np;  # Momentum step
    end
    # compared to collect(-(Pmax-dp/2):dp:(Pmax-dp/2))
    # this expression guarantees #points = 2*N_p
    (collect(-N_p:1.0:N_p-1) .+ 0.5) .*dp
end

function gen_Pzaxis(P_xm, P_ym, dpz::Float64)
    Pmax2 = maximum(abs.(P_xm))^2 + maximum(abs.(P_ym)) ^2;
    Pmin = min(minimum(abs.(P_xm)), minimum(abs.(P_ym)));
    Pzmax = sqrt(Pmax2 - Pmin^2);
    0:dpz:Pzmax
end

function get_Pgrid(Np::Int, Pz_Slice, dpz::Real)
    # A number of momentum points was specified
    paxis::Vector{Float64} = get_Paxis(Np; dp=dpz);
    P_ym = [j for i in 1:length(paxis) for j in paxis];
    P_xm = [i for i in paxis for j in 1:length(paxis)];
    P_z = (Pz_Slice==nothing) ? paxis[paxis.>=0] : Pz_Slice;
    return P_xm, P_ym, P_z
end

function get_Pgrid(Np::Array{T, 2}, Pz_Slice, dpz::Real) where {T<:Real}
    P_xm = @view Np[:,1]; 
    P_ym = @view Np[:,2]; 
    if Pz_Slice!=nothing
        P_z = Pz_Slice
    elseif size(Np,2) >= 3
        P_z = @view Np[:,3];
    else
        P_z = gen_Pzaxis(P_xm, P_ym, dpz)
    end
    return P_xm, P_ym, P_z
end

function get_Pgrid(Np::Vector{T}, Pz_Slice, dpz::Real) where {T<:Base.Generator}
    P_xm = Np[1]; 
    P_ym = Np[2]; 
    if Pz_Slice!=nothing
        P_z = Pz_Slice
    elseif length(tmps) >= 3
        P_z = Np[3];
    else
        P_z = gen_Pzaxis(P_xm, P_ym, dpz);
    end
    return P_xm, P_ym, P_z
end


function gen_PxyPolgrid(Pmax::Real, Npr::Int; Pmin::Real=0, Nth::Int=360)
    dpr::Float64 = (Pmax-Pmin)/Npr;
    praxis = ((i-0.5)*dpr +Pmin for i in 1:Npr);
    # println(praxis[[1,end]]);
    dq::Float64 = 2*pi / Nth;
    qaxis = 0:dq:2*pi-dq/2;
    grid = [(pr * f(q) for pr in praxis, q in qaxis) for f in [cos, sin]];
    # qaxis .*= 180/pi;
    collect(praxis), collect(qaxis).*(180/pi), grid
end

function checkPzslice(config::Dict{String, Any}; offset::Real=0)
    dpz = config["dpz"];
    if !(haskey(config, "Pz_slice"))
        Pzmax::Float64 = sqrt(2*(config["Kmax"] - get(config, "Kmin", 0.0)));
        config["Pz_slice"] = (offset*dpz):dpz:Pzmax;
    end
    length(config["Pz_slice"])
end

function checkAMphasors(config::Dict{String, Any})
    phases = collect(get(config, "AMphase", [0.0]));
    @. exp(1im * phases)
end

"""Cumulative trapz sum"""
function cumtrapz(A::AbstractArray{T}, out::AbstractArray{T}; dx::Array{T1,1}, dim::Int=1) where {T<:Number, T1<:Real}
    @assert size(out, dim) == length(dx)+1
    @assert size(out) == size(A)
    Ai = selectdim(A, dim, 1)
    newAi = selectdim(A, dim, 2)
    # mean = similar(Bi)
    mean = copy(Ai);
    lastmean = copy(newAi);
    mean .*= 0.5;
    lastmean .*= 0.5;
    selectdim(out, dim, 1) .= view(dx,1) .* mean;
    for i in 2:length(dx)
        Ai = selectdim(A, dim, i)
        newAi = selectdim(A, dim, i+1)
        @. mean = 0.5 * (Ai + newAi);
        selectdim(out, dim, i) .= selectdim(out,dim,i-1) .+ view(dx,i-1) .* lastmean
        @. lastmean = mean;
    end
    i = length(dx)+1
    selectdim(out, dim, i) .= selectdim(out,dim,i-1) .+ view(dx,i-1) .* lastmean
    return out
end
function cumtrapz(A::AbstractArray{T}; dx::Array{T1,1}, dim::Int=1) where {T<:Number, T1<:Real}
    out = copy(A);
    return cumtrapz(A, out; dx=dx, dim=dim)
end
"""In-place"""
function cumtrapz!(B::AbstractArray{T}; dx::Array{T1,1}, dim::Int=1) where {T<:Number, T1<:Real}
    return cumtrapz(B,B; dx=dx, dim=dim)
end
function cumtrapz!(B::AbstractArray{T}, x::Array{T1,1}; dim::Int=1) where {T<:Number, T1<:Real}
    dx = diff(x);
    return cumtrapz!(B; dx=dx,dim=dim)
end

"""Cumulative integration function, in-place"""
function cum_Integrate!(integrand::AbstractArray{T}, dt::Real; dim::Int=1)::Array{T} where {T<:Number}
    cumsum!(integrand, integrand, dims=dim);
    integrand .*= dt;
    return integrand
end
function cum_Integrate!(integrand::AbstractArray{T}, dt::Array{T1}; dim::Int=1)::Array{T} where {T<:Number, T1<:Real}
    cumtrapz!(integrand; dx=dt, dim=1)
end

"""Integration functions"""
function cum_Integrate(integrand::AbstractArray{T}, dt::Real; dim::Int=1)::Array{T} where {T<:Number}
    cumsum(integrand, dims=dim) * dt
end
function cum_Integrate(integrand::AbstractArray{T}, t_X::Array{T1}; dim::Int=1)::Array{T} where {T<:Number, T1<:Real}
    out = similar(integrand);
    cumtrapz!(out, integrand, t_X, dim=1)
end

function integrate(integrand::Base.Generator, dt::Real)
    dt * sum(integrand);
end

function integrate(integrand::AbstractArray{T}, dt::Real; dim::Int=1)::Array{T} where {T<:Number}
    dt .* sum(integrand, dims=dim)
end

# function integrate(integrand::AbstractArray{T}, t::Union{Vector{T1},StepRangeLen{T1}}; dim::Int=1)::Array{T} where {T<:Number, T1<:Real}
#     ret = fill(0.0,size(selectdim(integrand,dim,1)));
#     for i in 2:size(integrand, dim)
#         ret .+= (t[i]-t[i-1]).*(selectdim(integrand,dim,i).+selectdim(integrand,dim,i-1))
#     end
#     ret .*= 0.5
# end

function itrapz(integrand::AbstractArray{T}, t::Union{Vector{T1},StepRangeLen{T1}}; dim::Int=1)::Array{T} where {T<:Number, T1<:Real}
    ret = fill(0.0,size(selectdim(integrand,dim,1)));
    for i in 2:size(integrand, dim)
        ret .+= (t[i]-t[i-1]).*(selectdim(integrand,dim,i).+selectdim(integrand,dim,i-1))
    end
    ret .*= 0.5
end

function get_dtaxis(t_X::Vector{T}) where {T <: Real}
    difftX = diff(t_X);
    N_t::Int = size(t_X,1);
    if Stat.std(difftX)  < 1E-3 * abs(Stat.mean(difftX)) # Check if step size is uniform
        dt = (t_X[end]-t_X[1]) / ( N_t - 1 ); # println("Uniform t grid")
        tweight = Iterators.repeated(dt);
    else # Variable step size
        # dt = difftX; # println("Non-uniform t grid")
        tweight = cat(difftX,[0.],dims=1);
        tweight .+= cat(difftX[1:1],difftX,dims=1);
        tweight .*= 0.5;
        dt = tweight;
    end
    return (dt, tweight)
end

"""Dipole function"""
function dipole_M_H(Px::Real, P2::Real; Ip_au::Real=0.5, Py::Real=0, Pz::Real=0)::Real
    """This function calculates the dipole moment for ionization of a hydrogen
    atom. d = P ./ (P^2 + 2*Ip).^3;
    Px, Py, Pz: momentum components in a.u.
        Can be vectorized
    Ip_au: ionization potential in a.u.
    """
    # mesh of P^2 values
    local P2local = (P2 >= Px^2) ? P2 : Px^2+Py^2+Pz^2;
    # if P2 <= 0
    #     P2local = Px^2+Py^2+Pz^2;
    # else
    #     P2local = P2;
    # end
    # Prefactor
    F = 2^(7.0/2) * (2*Ip_au)^(5.0/4) / pi; 
    d = F * Px / ( P2local + 2*Ip_au )^3;
    d
end
# function dipole_M_H(Px, Py, 
#                     Pz, Ip_au::Real=0.5)::Real
#     """This function calculates the dipole moment for ionization of a hydrogen
#     atom. d = P ./ (P^2 + 2*Ip).^3;
#     Px, Py, Pz: momentum components in a.u.
#         Can be vectorized
#     Ip_au: ionization potention in a.u.
#     """
#     # mesh of P^2 values
#     P2 = Px^2+Py^2+Pz^2;
#     # Prefactor
#     F = 2^(7.0/2) * (2*Ip_au)^(5.0/4) / pi; 
#     d = F * Px / ( P2 + 2*Ip_au )^3;
#     d
# end

using Printf
f2str(f) = begin
out =  @sprintf("%dp", Int(div(f,1)));
out * rpad(@sprintf("%.2g", mod(f,1))[3:end], 1, '0')
end

