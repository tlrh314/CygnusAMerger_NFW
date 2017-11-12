import numpy


def apply_projection(EulAng, gas, dm)
    """ all vector quantities have to be rotated

        This method has been shamelessly copy-pasted from P-Smac2,
            see src/setup.c#L107-L208

        @param EulAng: three-vector with psi, theta, phi
        @param gas: AstroPy Table containing gas properties
        @param dm: AstroPy Table containing dm properties

        @return: gas, dm where the vectors have been rotated """

    YAWPITCHROLL = True
    deg2rad = numpy.pi / 180.0
    psi = EulAng[0] * deg2rad
    theta = EulAng[1] * deg2rad
    phi = EulAng[2] * deg2rad

    if not phi and not theta and not psi:
        return  # nothing to do

    A = numpy.zeros((3, 3))  # Rotation Matrix
    PartTotal = len(gas["x"])

    # Define rotation matrix
    if YAWPITCHROLL:  # Luftfahrtnorm (DIN 9300) (Yaw-Pitch-Roll, Z, Y’, X’’)
        A = [
                [
                    numpy.cos(theta) * numpy.cos(phi),
                    numpy.cos(theta) * numpy.sin(phi),
                    -numpy.sin(theta)
                ], [
                    numpy.sin(psi) * numpy.sin(theta) * numpy.cos(phi) - numpy.cos(psi) * numpy.sin(phi),
                    numpy.sin(psi) * numpy.sin(theta) * numpy.sin(phi) + numpy.cos(psi) * numpy.cos(phi),
                    numpy.sin(psi) * numpy.cos(theta)
                ], [
                    numpy.cos(psi) * numpy.sin(theta) * numpy.cos(phi) + numpy.sin(psi) * numpy.sin(phi),
                    numpy.cos(psi) * numpy.sin(theta) * numpy.sin(phi) - numpy.sin(psi) * numpy.cos(phi),
                    numpy.cos(psi) * numpy.cos(phi)
                ]
        ]
    else:  # Euler Matrix, y-Convention
       A = [
               [
                    -numpy.sin(psi) * numpy.sin(phi) + numpy.cos(psi) * numpy.cos(theta) * numpy.cos(phi),
                    -numpy.sin(psi) * numpy.cos(phi) - numpy.cos(psi) * numpy.cos(theta) * numpy.sin(phi),
                    numpy.cos(psi) * numpy.sin(theta)
                ], [
                    numpy.cos(psi) * numpy.sin(phi) + numpy.sin(psi) * numpy.cos(theta) * numpy.cos(phi),
                    numpy.cos(psi) * numpy.cos(phi) - numpy.sin(psi) * numpy.cos(theta) * numpy.sin(phi),
                    numpy.sin(psi) * numpy.sin(theta)
                ], [
                    -numpy.sin(theta) * numpy.cos(phi),
                    numpy.sin(theta) * numpy.sin(phi),
                    numpy.cos(theta)
                ]
            ]

    # Apply transformations to dm/gas positions
    x, y, z = dm["x"], dm["y"], dm["z"]
    dm["x"] = A[0][0] * x + A[0][1] * y + A[0][2] * z
    dm["y"] = A[1][0] * x + A[1][1] * y + A[1][2] * z
    dm["z"] = A[2][0] * x + A[2][1] * y + A[2][2] * z

    x, y, z = gas["x"], gas["y"], gas["z"]
    gas["x"] = A[0][0] * x + A[0][1] * y + A[0][2] * z
    gas["y"] = A[1][0] * x + A[1][1] * y + A[1][2] * z
    gas["z"] = A[2][0] * x + A[2][1] * y + A[2][2] * z

    # Apply transformations to dm/gas velocities
    vx, vy, vz = [dm["vx"], dm["vy"], dm["vz"]
    dm["vx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
    dm["vy"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
    dm["vz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz

    vx, vy, vz = [gas["vx"], gas["vy"], gas["vz"]
    gas["vx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
    gas["vy"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
    gas["vz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz

    # Apply transformations to magnetic field (gas only)
    if "Bx" in gas.keys():
        Bx, By, Bz = gas["Bx"], gas["By"], gas["Bz"]

        gas["Bx"] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
        gas["By"] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
        gas["Bz"] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];

    # Apply transformations to bulk velocity (gas only; only ifdef VTURB)
    if "VBulkx" in gas.keys():
        vx, vz, vz = gas["VBulkx"], gas["VBulky"], gas["VBulkz"]
        gas["VBulkx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
        gas["VBulky"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
        gas["VBulkz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz
