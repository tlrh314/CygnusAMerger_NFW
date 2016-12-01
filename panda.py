import numpy


def create_panda(xlen, ylen, xc, yc, r, a1, a2):
    """ Create mask of spherical wedge with radius r between angles a1 and a2
            for a numpy array of shape xlen, ylen
        @param xlen, ylen: shape of the array to mask
        @param xy, yc    : coordinates to use as center for the wedges
        @param r         : radius of the spherical shell
        @param a1, a2    : angles, where horizontal = 0; counterclockwise;
                           input values must be in degrees
        @return          : numpy mask containing indices
                           obtaining the masked values works as array[mask]
    """

    # Thickness of the spherical wedge shell section is one pixel now
    dr = 1

    # Create mask at center (xc, yc) for array.shape = (xlen, ylen)
    y,x = numpy.ogrid[-yc:ylen-yc, -xc:xlen-xc]

    # Convert degrees to radians
    a1 *= numpy.pi/180
    a2 *= numpy.pi/180

    # Some trial and error angle magic. Be careful with arctan2 domain!
    a1 = a1%(2*numpy.pi)
    a2 = a2%(2*numpy.pi)

    angles = numpy.arctan2(y,x)
    angles += 2*numpy.pi
    angles = angles % (2*numpy.pi)

    if a2 < a1:
        anglemask = ((a1 <= angles) | (angles <= a2))
    else:
        anglemask = ((a1 <= angles) & (angles <= a2))

    return (x**2 + y**2 <= r**2) & (x**2 + y**2 >= (r-dr)**2) & anglemask

