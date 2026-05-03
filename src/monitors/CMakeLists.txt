target_sources(ablateLibrary
        PRIVATE
        # fieldErrorMonitor.cpp
        # solutionErrorMonitor.cpp
        timeStepMonitor.cpp
        # ignitionDelayPeakYi.cpp
        # ignitionDelayTemperature.cpp
        # extractLineMonitor.cpp
        # dmViewFromOptions.cpp
        # particleCount.cpp
        # particleAverage.cpp
        curveMonitor.cpp
        # maxMinAverage.cpp
        # physicsTimeStep.cpp
        # probes.cpp
        # rocketMonitor.cpp
        # turbFlowStats.cpp
        # boundarySolverMonitor.cpp
        # fieldMonitor.cpp
        # mixtureFractionMonitor.cpp
        # mixtureFractionCalculator.cpp
        # radiationFieldMonitor.cpp
        # radiationFlux.cpp
        # chemTabMonitor.cpp

        PUBLIC
        monitor.hpp
        timeStepMonitor.hpp
        )

add_subdirectory(logs)
add_subdirectory(probes)