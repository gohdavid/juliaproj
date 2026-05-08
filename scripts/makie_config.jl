using CairoMakie

const PAPER_FONT = "Neue Haas Grotesk Display Pro"
const PAPER_FONTSIZE = 8
const PAPER_FIGSIZE_IN = (3.35, 1.675)
const PAPER_FIGSIZE_PT = round.(Int, 72 .* PAPER_FIGSIZE_IN)

const PAPER_COLORS = [
    colorant"#0072B2",
    colorant"#E69F00",
    colorant"#009E73",
    colorant"#CC79A7",
    colorant"#56B4E9",
    colorant"#D55E00",
    colorant"#F0E442",
    colorant"#000000",
]

const PAPER_LINESTYLES = [
    :solid,
    :dash,
    :dashdot,
    :dot,
]

const PAPER_THEME = Theme(
    font=PAPER_FONT,
    fontsize=PAPER_FONTSIZE,
    linewidth=1.5,
    palette=(;
        color=PAPER_COLORS,
        linestyle=PAPER_LINESTYLES,
    ),
    Lines=(;
        cycle=Cycle([:color, :linestyle], covary=false),
        linewidth=1.5,
    ),
    ScatterLines=(;
        cycle=Cycle([:color, :linestyle], covary=false),
        linewidth=1.5,
    ),
    Figure=(;
        size=PAPER_FIGSIZE_PT,
        backgroundcolor=:white,
    ),
    Axis=(;
        backgroundcolor=:white,
        spinewidth=1,
        leftspinecolor=:black,
        rightspinecolor=:black,
        topspinecolor=:black,
        bottomspinecolor=:black,
        rightspinevisible=false,
        topspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
        titlesize=PAPER_FONTSIZE,
        xlabelsize=PAPER_FONTSIZE,
        ylabelsize=PAPER_FONTSIZE,
        xtickalign=0,
        ytickalign=0,
        xticksize=3,
        yticksize=3,
        xtickwidth=1,
        ytickwidth=1,
        xticklabelsize=PAPER_FONTSIZE,
        yticklabelsize=PAPER_FONTSIZE,
        xticklabelpad=3,
        yticklabelpad=3,
    ),
    Legend=(;
        labelsize=PAPER_FONTSIZE,
        framevisible=false,
    ),
)

function set_paper_theme!()
    set_theme!(PAPER_THEME)
end

function paper_figure(; size=PAPER_FIGSIZE_PT, kwargs...)
    return Figure(; size, backgroundcolor=:white, kwargs...)
end

function save_paper_pdf(path::AbstractString, fig::Figure; kwargs...)
    return save(path, fig; pt_per_unit=1, kwargs...)
end
