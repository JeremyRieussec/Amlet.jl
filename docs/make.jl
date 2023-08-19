using Documenter
using Amlet

makedocs(
    sitename = "Amlet",
    format = Documenter.HTML(),
    modules = [Amlet]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JeremyRieussec/Amlet.jl";
    push_preview = true
)
