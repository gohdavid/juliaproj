using Dates
using SHA

export load_yaml_config, write_yaml_config, cfgget, cfgsymbol, cfgbool, cfgint, cfgfloat, cfgfloat32
export config_hash, config_output_dir

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
indentation plus scalar values and bracket arrays. A top-level `extends` key
loads another config relative to this file and deep-merges this file over it.
"""
function _load_yaml_config_flat(path::AbstractString)
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

function _deep_merge_config(base::AbstractDict, override::AbstractDict)
    result = ConfigDict()
    for (key, value) in base
        result[String(key)] = value isa AbstractDict ?
                              _deep_merge_config(value, ConfigDict()) :
                              value
    end
    for (key, value) in override
        key_str = String(key)
        key_str == "extends" && continue
        if haskey(result, key_str) &&
           result[key_str] isa AbstractDict &&
           value isa AbstractDict
            result[key_str] = _deep_merge_config(result[key_str], value)
        else
            result[key_str] = value
        end
    end
    return result
end

function _config_parent_path(path::AbstractString, parent::AbstractString)
    return isabspath(parent) ? parent : joinpath(dirname(path), parent)
end

function load_yaml_config(path::AbstractString; _seen::Vector{String}=String[])
    resolved_path = abspath(path)
    if resolved_path in _seen
        chain = join(vcat(_seen, resolved_path), " -> ")
        error("Cyclic YAML extends chain: $chain")
    end

    cfg = _load_yaml_config_flat(resolved_path)
    parent = get(cfg, "extends", nothing)
    parent === nothing && return cfg

    parent_cfg = load_yaml_config(_config_parent_path(resolved_path, String(parent));
                                  _seen=vcat(_seen, resolved_path))
    return _deep_merge_config(parent_cfg, cfg)
end

function _write_yaml_value(io::IO, key::AbstractString, value, indent::Int)
    prefix = repeat(" ", indent)
    if value isa AbstractDict
        println(io, prefix, key, ":")
        for child_key in sort!(collect(keys(value)); by=string)
            _write_yaml_value(io, String(child_key), value[child_key], indent + 2)
        end
    elseif value isa AbstractVector
        println(io, prefix, key, ": [", join(string.(value), ", "), "]")
    elseif value === nothing
        println(io, prefix, key, ": null")
    elseif value isa AbstractString
        println(io, prefix, key, ": ", value)
    else
        println(io, prefix, key, ": ", value)
    end
end

function write_yaml_config(path::AbstractString, cfg::AbstractDict)
    open(path, "w") do io
        for key in sort!(collect(keys(cfg)); by=string)
            _write_yaml_value(io, String(key), cfg[key], 0)
        end
    end
    return path
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

function _canonical_config(value)
    if value isa AbstractDict
        parts = String[]
        for key in sort!(collect(keys(value)); by=string)
            push!(parts, repr(String(key)) * ":" * _canonical_config(value[key]))
        end
        return "{" * join(parts, ",") * "}"
    elseif value isa AbstractVector
        return "[" * join((_canonical_config(item) for item in value), ",") * "]"
    elseif value === nothing
        return "nothing"
    elseif value isa AbstractString
        return repr(String(value))
    elseif value isa Symbol
        return repr(String(value))
    else
        return repr(value)
    end
end

function _drop_excluded(cfg::AbstractDict, exclude)
    excluded = Set(String.(exclude))

    function drop_value(value, prefix::String)
        if value isa AbstractDict
            result = Dict{String,Any}()
            for (key, child) in value
                key_str = String(key)
                path = isempty(prefix) ? key_str : prefix * "." * key_str
                path in excluded && continue
                result[key_str] = drop_value(child, path)
            end
            return result
        end
        return value
    end

    return drop_value(cfg, "")
end

function config_hash(cfg::AbstractDict; n_chars::Int=12, exclude=())
    n_chars >= 3 || error("config_hash needs at least 3 characters")
    digest = bytes2hex(sha256(_canonical_config(_drop_excluded(cfg, exclude))))
    return digest[begin:min(end, n_chars)]
end

function config_output_dir(cfg::AbstractDict; default_root::AbstractString="outputs",
                           default_name::AbstractString="experiment")
    root = String(cfgget(cfg, "output.dir", default_root))
    name = String(cfgget(cfg, "output.experiment_name",
                         cfgget(cfg, "experiment.name", default_name)))
    digest = config_hash(cfg; exclude=(
        "output",
        "training.epochs",
        "training.checkpoint_every",
        "training.resume_checkpoint",
    ))
    return joinpath(root, name, digest)
end
