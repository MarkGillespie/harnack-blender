# Harnack Tracing in Blender
This repository contains our Blender implementation of the algorithm described in _Ray Tracing Harmonic Functions_ for ray tracing harmonic functions.

## Building
Here are the commands that I use to build this code on a Mac. First, Blender requires svn (which can be installed using e.g. `brew install svn`).

Then, you can run the following commands to build and launch Blender
```bash
mkdir blender
cd blender
git clone git@github.com:MarkGillespie/harnack-blender.git blender-git
cd blender-git
make update
make -j11
../build_darwin/bin/Blender.app/Contents/MacOS/Blender
```

## Code
Most of our new code can be found in `intern/cycles/kernel/geom/nonplanar_polygon_intersect.h` and `intern/cycles/kernel/geom/harnack.ipp`

## Acknowledgements
Elliptic integrals are evaluated using code by John Burkardt, available [here](https://people.math.sc.edu/Burkardt/f77_src/elliptic_integral/elliptic_integral.html).
 

## Original Blender README
<!--
Keep this document short & concise,
linking to external resources instead of including content in-line.
See 'release/text/readme.html' for the end user read-me.
-->

Blender
=======

Blender is the free and open source 3D creation suite.
It supports the entirety of the 3D pipeline-modeling, rigging, animation, simulation, rendering, compositing,
motion tracking and video editing.

![Blender screenshot](https://code.blender.org/wp-content/uploads/2018/12/springrg.jpg "Blender screenshot")

Project Pages
-------------

- [Main Website](http://www.blender.org)
- [Reference Manual](https://docs.blender.org/manual/en/latest/index.html)
- [User Community](https://www.blender.org/community/)

Development
-----------

- [Build Instructions](https://wiki.blender.org/wiki/Building_Blender)
- [Code Review & Bug Tracker](https://projects.blender.org)
- [Developer Forum](https://devtalk.blender.org)
- [Developer Documentation](https://wiki.blender.org)


License
-------

Blender as a whole is licensed under the GNU General Public License, Version 3.
Individual files may have a different, but compatible license.

See [blender.org/about/license](https://www.blender.org/about/license) for details.
