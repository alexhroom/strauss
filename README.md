# strauss
Python code for Chapter 8 of [Partial Differential Equations, An Introduction by Walter A. Strauss.](https://www.wiley.com/en-us/Partial+Differential+Equations%3A+An+Introduction%2C+2nd+Edition-p-9780470054567)

Chapter 8 of this textbook concerns Computation of Solutions. It consists of:
- 8.1: Opportunities and Dangers  
  Introduces the finite differences method and shows the stability 
  issues created by a poor choice of time-step.
- 8.2: Approximations of Diffusions  
  Fixes the time-step problem, uses finite differences to approximate a simple
  diffusion problem and discusses stability. Also introduces how to computationally
  implement Neumann boundary conditions, and the Crank-Nicolson scheme.
- 8.3: Approximations of Waves
- 8.4: Approximations of Laplace's Equation
- 8.5: Finite Element Method

This is mostly discussed in theoretical detail, and how to compute these "by hand". I have written Python code to accompany
the examples and concepts in the book, with graphs and animations.
