export MNIST
module MNIST

using GZip
using FixedPointNumbers

const defdir = joinpath(Pkg.dir("MLDatasets"), "datasets/mnist")

macro dataset(name, image_file, label_file)
    quote
        function ($name){T}(::Type{T}, dir=defdir)
            xraw = data(UFixed8, dir, $image_file)
            yraw = data(UInt8, dir, $label_file)
            x = convert(Array{T}, xraw)
            y = convert(Array{Int}, yraw)
            x, y
        end

        ($name)(dir=defdir) = ($name)(Float64, defdir)
    end |> esc
end

@dataset traindata "train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz"
@dataset testdata "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz"

function data{T}(::Type{T}, dir, filename)
    mkpath(dir)
    url = "http://yann.lecun.com/exdb/mnist"
    path = joinpath(dir, filename)
    isfile(path) || download("$(url)/$(filename)", path)
    stream = gzopen(path)
    magic_number = ntoh(read(stream, Int32))
    @assert magic_number == 2049 || magic_number == 2051
    ndims = magic_number & 0xff
    dim = reverse(ntoh.(read(stream, Int32, ndims)))
    data = read(stream, T, dim...)
    close(stream)
    return data
end

end
