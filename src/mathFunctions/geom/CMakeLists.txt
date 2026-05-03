target_sources(ablateLibrary
        PRIVATE
        geometry.cpp
        sphere.cpp
        box.cpp
        # surface.cpp  # disabled: depends on EGADS/OpenCASCADE; see root CMakeLists note
        cylinderShell.cpp
        cylinder.cpp
        union.cpp
        difference.cpp
        inverse.cpp
        triangle.cpp
        convexPolygon.cpp

        PUBLIC
        geometry.hpp
        sphere.hpp
        box.hpp
        # surface.hpp  # disabled: depends on EGADS/OpenCASCADE
        cylinderShell.hpp
        cylinder.hpp
        union.hpp
        difference.hpp
        inverse.hpp
        triangle.hpp
        convexPolygon.hpp
        )