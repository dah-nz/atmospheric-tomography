
def palette(titles=None):
    colour_list = [ #"#172F4FFF",
                    "#0C2340FF",
                    "#D83131FF",
                    "#FFC042FF",
                    "#2E86ABFF",
                    "#8CD867FF",
                    "#00000080",
                    "#8CD86780"]
    if titles is not None:
        return dict(zip(titles, colour_list))
    else:
        return colour_list