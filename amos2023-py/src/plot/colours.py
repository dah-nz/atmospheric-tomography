
def palette(titles=None):
    colour_list = [ "#172F4FFF",
                    "#D83131FF",
                    "#FFC042FF",
                    "#8CD867FF",
                    "#2E86ABFF",
                    "#8CD86780"]
    if titles is not None:
        return dict(zip(titles, colour_list))
    else:
        return colour_list