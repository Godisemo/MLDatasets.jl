export MNIST
module MNIST

using GZip

const defdir = joinpath(Pkg.dir("MLDatasets"), "datasets/mnist")

"""
* [dir]: save directory. Default: "MLDatasets/datasets/mnist"
"""
function traindata(dir=defdir)
    x = convert(Array{Float64}, data(dir, "train-images-idx3-ubyte.gz")) / 255
    y = convert(Array{Int64}, data(dir, "train-labels-idx1-ubyte.gz"))
    x, y
end

"""
* [dir]: save directory. Default: "MLDatasets/datasets/mnist"
"""
function testdata(dir=defdir)
    x = convert(Array{Float64}, data(dir, "t10k-images-idx3-ubyte.gz")) / 255
    y = convert(Array{Int64}, data(dir, "t10k-labels-idx1-ubyte.gz"))
    x, y
end

function data(dir, filename)
    mkpath(dir)
    url = "http://yann.lecun.com/exdb/mnist"
    path = joinpath(dir, filename)
    isfile(path) || download("$(url)/$(filename)", path)
    stream = gzopen(path)
    magic_number = ntoh(read(stream, Int32))
    @assert magic_number == 2049 || magic_number == 2051
    ndims = magic_number & 0xff
    dim = reverse(ntoh.(read(stream, Int32, ndims)))
    data = read(stream, UInt8, dim...)
    close(stream)
    return data
end

end
