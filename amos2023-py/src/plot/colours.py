
def palette(titles=None):
    # colour_list = [ "#172F4FFF",
    #                 "#D83131FF",
    #                 "#FD7F20FF",
    #                 "#FFC042FF",
    #                 "#2E86ABFF",
    #                 "#8CD867FF",
    #                 "#53B3CBFF",
    #                 "#781717FF"]
    colour_list = [
        "#091F34",
        "#D83131",
        "#E3B505",
        "#0091AD",
        "#BD93BD",
        "#52AA5E",
        "#DE6E4B"
    ]
    if titles is not None:
        return dict(zip(titles, colour_list))
    else:
        return colour_list