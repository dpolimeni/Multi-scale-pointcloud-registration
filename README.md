# Registration Library OR-PCD 🚀
☁ OR-PCD is a library that implements a two block optimization approach in order to align Pointclouds. This method was created as a solution capable to manage scale and differences among the three coordinate axes of the clouds. In our approach we optimize the scale differences considering all the x/y/z coordinates directions as variables.

It integrates an inner optimization block where standard rototranslation matrix is obtained and an outer one where the best scale factors are estimated through with a pattern search algorithm ⠻⠏.

# Library Structure
The library is composed of 3 main components and the objective is the let them be as extendible as possible in order to add other optimization pipelines during time.
These are the components:
- ⛓ Preprocessor: the preprocessor represent the standardization pipeline used before aligning the pointclouds. At the moment is composed of:
  1. Downsamplers: Porcess blocks that downsample pointclouds having too much points
  2. Scalers: Process block that scale the clouds initially
  3. Outliers: Porcess blocks that let you remove outliers from the cloud
- 🔝 Optimizer: Inner block optimization methods. At the moment FastGlobal/GeneralizedICP are available
- ⁑ The Aligner: is the outer block of the optimization. At the moment not extendible as it performs a pattern search with a multi-start procedure in order to find best scale parameters.

## Sample Data
We provide some sample data in our libary importable.

# Equally involved contributors:

- Diego Polimeni: diego.polimeni99@gmail.com
- Alessandro Pannone: a.pannone1798@gmail.com

Feel free to reach out to the contributors if you have any questions or suggestions.

Thank you for your interest in our project!
