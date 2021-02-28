The aerospace dataset is based on an in-house 2D wing simulation based on  flat-plate pitch-plunge model, with the two control surfaces modeled. Attached to the 2D wing are two control surfaces, and five sensors:

 - Two sensors are on the wing tip
 - Two sensors are at wing mid-span
 - One sensor is on the first control surface, where damage will be introduced

The purpose of aerospace dataset is to represent a situation that has:

 1. **Low data cardinality:** In expensive, dynamical systems one does not have the luxury to take measurements at will
 2. **Multiple transitionary phases:** Note that an aircraft will experience various flight phases during operation. In an naive anomaly detection algorithm, each flight phase transition *will* be classed as an anomaly to one another. Therefore there is a need to pin-point the difference(s) between normal flight transitions **and** when damage has occured, which requires a sophisticated look at anomaly detection.

The following transitions are modeled:

 1. Take-off phase
 2. Climb phase
 3. Climb phase with damage

based on an assumed **single** flight regime (recall that test flights are expensive to organize, therefore we challenge the analysis for low data cardinality analysis).
