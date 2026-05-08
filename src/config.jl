using Dates

export load_yaml_config, cfgget, cfgsymbol, cfgbool, cfgint, cfgfloat, cfgfloat32

const ConfigDict = Dict{String,Any}

function _strip_comment(line::AbstractString)
    in_quote = false
    quote_char = '\0'
    for (i, c) in pairs(line)
        if c == '"' || c == '\''
            if !in_quote
                in_quote = true
                quote_char = c
            elseif quote_char == c
                in_quote = false
            end
        elseif c == '#' && !in_quote
            return rstrip(line[begin:prevind(line, i)])
        end
    end
    return rstrip(line)
end

function _parse_scalar(raw::AbstractString)
    s = strip(raw)
    isempty(s) && return ""
    if (startswith(s, "\"") && endswith(s, "\"")) ||
       (startswith(s, "'") && endswith(s, "'"))
        return s[2:end-1]
    elseif s in ("true", "True", "yes", "on")
        return true
    elseif s in ("false", "False", "no", "off")
        return false
    elseif s in ("nothing", "null", "None", "~")
        return nothing
    elseif startswith(s, "[") && endswith(s, "]")
        inner = strip(s[2:end-1])
        isempty(inner) && return Any[]
        return Any[_parse_scalar(part) for part in split(inner, ",")]
    end

    parsed_int = tryparse(Int, s)
    parsed_int !== nothing && return parsed_int
    parsed_float = tryparse(Float64, s)
    parsed_float !== nothing && return parsed_float
    return s
end

"""
    load_yaml_config(path)

Load the small YAML subset used by experiment configs: nested maps via
indentation plus scalar values and bracket arrays. This keeps experiment startup
dependency-light while still allowing reproducible config files.
"""
function load_yaml_config(path::AbstractString)
    root = ConfigDict()
    stack = Tuple{Int,ConfigDict}[(-1, root)]

    for raw in eachline(path)
        line = _strip_comment(raw)
        isempty(strip(line)) && continue
        indent = length(line) - length(lstrip(line))
        stripped = strip(line)
        occursin(":", stripped) || error("Invalid YAML line in $path: $raw")

        key, value = split(stripped, ":"; limit=2)
        key = strip(key)
        value = strip(value)

        while indent <= stack[end][1]
            pop!(stack)
        end
        parent = stack[end][2]

        if isempty(value)
            child = ConfigDict()
            parent[key] = child
            push!(stack, (indent, child))
        else
            parent[key] = _parse_scalar(value)
        end
    end

    return root
end

function cfgget(cfg::AbstractDict, path::AbstractString, default=nothing)
    cur = cfg
    for key in split(path, ".")
        if cur isa AbstractDict && haskey(cur, key)
            cur = cur[key]
        else
            return default
        end
    end
    return cur
end

cfgsymbol(cfg::AbstractDict, path::AbstractString, default) =
    Symbol(cfgget(cfg, path, default))
cfgbool(cfg::AbstractDict, path::AbstractString, default) =
    Bool(cfgget(cfg, path, default))
cfgint(cfg::AbstractDict, path::AbstractString, default) =
    Int(cfgget(cfg, path, default))
cfgfloat(cfg::AbstractDict, path::AbstractString, default) =
    Float64(cfgget(cfg, path, default))
cfgfloat32(cfg::AbstractDict, path::AbstractString, default) =
    Float32(cfgget(cfg, path, default))
