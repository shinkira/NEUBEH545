%-------------------------------------------------------------
% Linear Algebra Tutorial for NeuBeh/PBIO 545
%
% This tutorial introduces some basic concepts from linear algebra.  The aims are: (1) learn to visualize operations in
% linear algebra (e.g. different types of matrix transformations) (2) introduce some key types of linear algebra
% transformations and their properties (3) identify 'natural' coordinate systems(eigenvectors) to describe a system (4)
% illustrate these concepts with some real examples
%
% If you need a review of the basic concepts in linear algebra (vectors and matrices) read the Linear Algebra Primer from
% Eero Simoncelli that's on the course website.
%
% If you need an introduction to the basics of matlab (how to index into vectors and matrices, etc) see the links on the
% course website.
%
% Created 2/03 Fred Rieke 
% Revised 12/04 FMR
%               cleaned up for 2005 course.  
% Revised 12/06 FMR
%               more cleaning for 2007
% Revised 12/08 FMR
%               cleaned up, put in cells for 2009
%
% Dependencies:
%   You will need PlotVector.m and Plot3DVector.m to run the tutorial.
%-------------------------------------------------------------

%%
% default plotting stuff
PlotColors = 'bgrkymcbgrkymc';
set(0, 'DefaultAxesFontName','Helvetica')
set(0, 'DefaultAxesFontSize', 14)
%%

%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 1: Representation and properties of vectors
%
% Linear algebra relies on vectors and matrices.  Vectors are simply ordered collections of points.  The points are ordered
% so that a correspondence can be made between the value and a particular direction or axis.  You are all familiar with this
% from plotting points in cartesian (x-y-z) coordinates, where you can represent a point as (x,y,z).  This is a general
% concept.  For example, Starbucks could choose to represent different types of espresso drinks as collections of two numbers
% - one for the amount of milk and the other for the amount of espresso. (This is at least how it should be - i.e. before you
% could choose different types of milk, different flavors, ... .  Those would add more axes):

cap = [0.5
	   0.5];
espresso = [0
            1];
latte = [0.7
         0.3];

% These are column vectors.  We could also use row vectors - e.g. cap = [0.5 0.5].  We will mostly use column vectors.  The
% difference between row and column vectors is important mainly when we start thinking about matrices (below).

% The advantage of using vectors to represent data is that it allows us to visualize the data (provided it has low enough
% dimension).  Also, as we will see below, we can perform mathematical operations on vectors using matrices - in many
% instances this very efficient and even intuitive.

% Let's take a look at each vector above
figure(1);
clf;
PlotVector(cap, 'b')
hold on;
PlotVector(espresso, 'r')
PlotVector(latte, 'k')
xlabel('milk fraction');
ylabel('espresso fraction');
legend('cap', 'espresso', 'latte');
hold off;

% (If Matlab won't run the above lines make sure PlotVector is on your Matlab path - try typing 'which PlotVector' in the
% command window if you are not sure).  

% One thing this example points out is that the units on the axes do not need to be the same.  It's also good for small talk
% next time you are at Starbucks.  

% Matrices are operators that transform one vector to another.  They can scale the data along one or more axes, rotate the
% data to choose another set of axes, project the data along a subset of the axes, project from one space (e.g.
% milk-espresso) to another (sleep-wake).  The rest of the tutorial is devoted to these transformations.
%%

%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 2: n equations and n unknowns and matrix inverse
%
% One use of linear algebra, usually the first you learn in a real linear algebra course, is to solve systems of linear
% equations.  For example, what combination (x,y,z) satisfies
%	2x + 3y - 3z = 1 
%   -x + y = 2 
%   3x - y + z = -1
% These are linear because x, y, and z occur to power one (i.e. there are no x^2 or x^0.5, etc, terms) and are not multiplied
% by each other (no xy, yz, etc terms).

% Coupled linear equations have a solution as long as there are as many equations as unknowns (3 of each in our case) and
% provided the equations are independent (i.e. we cannot get one equation as a weighted sum of the other two).  This is the
% case above.  When either of these assumptions fails there is either no exact solution to the set of equations, or their are
% multiple solutions.  All is not lost necessarily though, as we often can find an approximate solution (using e.g. singular
% value decomposition, see below).

% One approach to solving this set of equations for x, y and z is to solve one equation for one of the variables and
% substitute into the other two equations.  If we do this twice we eliminate two of the variables and get a simple algebraic
% equation for the last.  Then we can work backwards to solve for the other two.  

% A second (more efficient) way to solve this is to use linear algebra.
% We rewrite our set of linear equations above as a matrix equation 
%       M u = v,
% where M is a 3x3 matrix, and u and v are 1x3 vectors:

M = [2 3 -3
	 -1 1 0
	 3 -1 1];
v = [1
	 2
	 -1];

% u is our unknown vector (containing x, y and z).  Make sure you
% understand this notation - i.e. how this is equivalent to the 3 equations
% above.  If this is obscure, consider the simpler case: if M = [2 0;0 3] and
% v = [4; 6] what would u be?

% What we want to do is find the vector u that satisfies Mu = v. We can
% think about this process visually as the following.  v is a vector in a 3
% dimensional space:

figure(1)
clf;
Plot3DVector(v, 'b');
xlabel('x');
ylabel('y');
zlabel('z');

% M is an operator in this space that transforms one vector to another.
% And our problem is to find the vector u that gets transformed by M into
% v.  Let's look at some examples:

%%
figure(1)
u = [1
	 0
	 0];
Plot3DVector(v, 'b');
hold on
Plot3DVector(u, 'g');
Plot3DVector(M * u, 'r');
hold off	 
xlabel('x');
ylabel('y');
zlabel('z');
legend(' v ', ' u ', ' Mu ')

% Our initial guess u is in green, M*u is in red, and our desired vector v
% is in blue.  We are a long way off. Here's another attempt:
%%

figure(1)
u = [0
	 2
	 1];
Plot3DVector(v, 'b');
hold on
Plot3DVector(u, 'g');
Plot3DVector(M * u, 'r');
hold off	 
xlabel('x');
ylabel('y');
zlabel('z');
legend(' v ', ' u ', ' Mu ')

% That's closer - but I also cheated to guess a form for u by looking at
% the answer first.  At this point you are probably ready to abandon the
% whole linear algebra approach.  If not try other guesses to convince
% yourself this is not a fruitful approach! So we'd like a more systematic
% approach to finding the appropriate u. This is what the matrix inverse
% will do for us.  If M is a matrix that tranforms a vector u to a vector
% v, inv(M) undoes that transformation --- i.e. it takes v back into u.

%%

% First let's make sure that the inv(M) works as advertised.  Starting with
% a vector u, we will apply M to create a new vector v, then we will apply
% inv(M) to v and make sure we get u back:

u = [1
	 1
	 1];
figure(1)
subplot(1, 3, 1)
Plot3DVector(u, 'g');
axis([0 3 0 3 0 3])
subplot(1, 3, 2)
Plot3DVector(M * u, 'r');
axis([0 3 0 3 0 3])
xlabel('x');
ylabel('y');
zlabel('z');
subplot(1, 3, 3)
Plot3DVector(inv(M) * M * u, 'k');
axis([0 3 0 3 0 3])

% On the left is our original u, in the middle is M*u and on the right is
% inv(M)*M*u. In other words, inv(M) * M transforms a vector back into
% itself.  You can see this from the structure of inv(M) * M itself

inv(M) * M

% This is the identity matrix - 1's along the diagonal and 0's everywhere
% else.  The identity simply transforms a vector back into itself.

% Everything above assumes the inverse exists, which we will return to below.

%%
% What does all this have to do with our 3 equations and 3 unknowns problem
% above?  Well, using the inverse we can repose the problem 
%       M u = v 
% by multiplying each side of the equation by inv(M).  Now we get 
%       inv(M) M u = inv(M) v.  
% But since inv(M) M u is just u, this is just 
%       u = inv(M) v.
% This helps a lot because now the problem is to find the vector that v is
% transformed into by inv(M). This transformed vector, u, is the vector
% we were looking for originally and is the same set of values of x,y,z
% that we get by brute force substitution in the original set of equations:

u = inv(M) * v

% check - M * u should be v

M * u

%%
%-------------------------------------------------------------
% Question set #1:
%	(1) Why does the identity transform a vector back into itself?  Pick
%       some examples and convince yourself this is true. 
%   (2) Confirm that the u we get above is right by the brute force approach
%		to the original coupled linear equations.
%	(3) Why does checking that M*u = v tell us we found the right solution
%       u?
%-------------------------------------------------------------

%%
%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 3: Examples of invertible matrix transformations
%
%	In this section we'll cover three types of invertible transformations: 
% reflection, scaling and rotation matrices. Reflection, scaling and
% rotation are examples of linear operators, which must satisfy two
% conditions.  First is additivity:
%       L(u + v) = L(u) + L(v)
% i.e. the operator applied to the sum of two vectors is equal to the sum
% of the operator applied to each vector.  Second is homogeneity:
%       L(cu) = cL(u)
% where c is a scalar constant.  

% Invertible transformations are, as the name suggests, transformations
% that can be undone.  This means that the matrix M generating the original
% transformation has an inverse.  In the next section we'll look at
% non-invertible transformations. 

%-------------------------------------------------------------
% Rotation matrices.
%	One important class of invertible transformations is rotations.
% Rotation matrices preserve the length of a vector but change its
% direction.  They do what the name implies: they rotate a vector by some
% specified angle, about some axis. A given rotation matrix will rotate
% every vector through the same angle.  

% Rotations in 2d
% 	Let's start with a vector in a 2-d space

figure(1)
clf
v = [0.5
	 0.8]
PlotVector(v, 'b')
xlabel('x component')
ylabel('y component')
axis([-1 1 -1 1])
grid on

%%
% Now let's rotate this vector by 40 degrees - this corresponds to shifting
% the phase of our signal by 40 degrees:

Angle = 2 * 3.14159 * 40/360;		% 40 degree angle expressed in steradians (2*PI = 360 degrees)
RotationMatrix = [cos(Angle) -sin(Angle)
				  sin(Angle) cos(Angle)];

% check out rotated vector
Rotatedv = RotationMatrix * v;
figure(1)
plot(0)
hold on
PlotVector(v, 'b');				% original
xlabel('x component')
ylabel('y component')
grid on
axis([-1 1 -1 1])
pause(1)
PlotVector(Rotatedv, 'r');		% rotated
hold off

%%
% This is not restricted to 2d.  Here's an example of a rotation 
% in a 3d space, where we rotate about the z axis:

v = [1
	 -0.5
	 1];
Angle = 2 * 3.14159 * 40/360;		
RotationMatrix = [cos(Angle) -sin(Angle) 0
				  sin(Angle) cos(Angle)  0
				  0				0   	 1];

% Look at the structure of the rotation matrix.  The upper left elements
% look just like our 2-d rotation matrix.  So the action of this matrix on
% the x and y coordinates of the vector will be the same as in the 2-d
% case.  What about the z-axis?  The 0s in the x and y spots in both the
% last column of the matrix mean that the x and y components of the vector
% do not pick up parts of the z component when the matrix is applied (if
% you don't see this write out what the matrix multiplication looks like).
% The 0s in the last row make sure that the z components of the vector do
% not pick up parts of the x and y.  Finally the 1 preserves the length of
% the z component.  Hence this is rotation about the z axis.
				  				  
Rotatedv = RotationMatrix * v;
figure(2)
clf
Plot3DVector(v, 'b');			% original
axis([-1 1 -1 1 -1 1])
pause(1)
hold on
Plot3DVector(Rotatedv, 'r');	% rotated
axis([-2 2 -2 2 -2 2])
hold off
xlabel('x');
ylabel('y');
zlabel('z');

%%
% If we rotate the axis so we are looking straight down the z axis we can see what happened in 
% the x-y plane:
figure(2)
view(0,90)

%%
% Rotation matrices are invertible - if we rotate by 40 degrees
% and then by -40 degrees we end up with the original vector:
v = [1
	 -0.5
	 1];

Angle = 2 * 3.14159 * (-40/360);
InvRotationMatrix = [cos(Angle) -sin(Angle) 0
				     sin(Angle) cos(Angle)  0
				     0				0   	1];
Rotatedv = RotationMatrix * v;
InvRotatedv = InvRotationMatrix * Rotatedv;
figure(2)
clf
subplot(1, 3, 1)
Plot3DVector(v, 'b');				% original
axis([0 1 0 1 0 1])
subplot(1, 3, 2)
Plot3DVector(Rotatedv, 'r');		% rotated
axis([0 1 0 1 0 1])
xlabel('x');
ylabel('y');
zlabel('z');
subplot(1, 3, 3)
Plot3DVector(InvRotatedv, 'g');		% rotated and inverse-rotated
axis([0 1 0 1 0 1])

%%
% This is a case where we know how to construct the inverse explicitly.
% Typically we'll have to find the inverse, using, e.g. the inv operation
% in Matlab.  As a sanity check, compare the -40 degree rotation matrix to
% the inverse of our original 40 degree rotation matrix
inv(RotationMatrix)
InvRotationMatrix

% and the matrix product should give us the identity matrix

RotationMatrix * inv(RotationMatrix)

%%
%-------------------------------------------------------------
% Reflection matrices
%
% Reflection also preserved length.  As a simple example, consider reflection
% about the y-axis:

figure(1)
clf
hold on
PlotVector([0.3 0.9], 'b')
PlotVector([0.3 -0.9], 'r')
grid
axis([-1 1 -1 1])
pause(1)
PlotVector([-0.3 0.9], 'b')
PlotVector([-0.3 -0.9], 'r')
hold off

% In each case the red vector is identical to the blue except it's y-axis
% coordinate has been multiplied by -1. Convince yourself that rotation
% through a single angle cannot account for the changes in both vectors.
% As a hint make sure you consider the sign of your rotation angle.

%%
%-------------------------------------------------------------
% Scaling matrices
%
% Scaling matrices stretch or contract one or more axes of the coordinate
% system, but do not mix the axes as a rotation matrix does.  The indentity
% matrix is our first (and particularly boring) example of a scaling matrix
% - it just happens to apply a scale of 1 to every axis.

% Here's a better example.  We have a two-dimensional distribution of data points,
% characterized by the mean and standard deviation along the x and y axis:

clear Dist;
Dist(1, :) = normrnd(0,0.5,1, 5000);
Dist(2, :) = normrnd(0,1.5,1, 5000);

% Thus the 2x5000 matrix Dist contains 5000 data points, each of which is an (x,y) pair.
% The distribution as a whole is characterized by the mean and standard deviation 
% along x and y (both means are 0, standard deviations are 0.5 and 1.50.  
% Each (x,y) pair of values can be represented as a 2-d vector.  To keep the plot
% simple, we'll plot only the ends of the vectors (i.e. the data points themselves):

figure(1)
clf
subplot(1, 2, 1)
plot(Dist(1, :), Dist(2, :), '.');
axis([-5 5 -5 5])
axis square
xlabel('amp 1')
ylabel('amp 2')

%%
% Let's say we want to measure distance of a given data point from the
% origin (which is also the mean of the distribution), but we want to measure distance in
% standard deviations.  That is, given a point (x,y,) we want to know how
% many standard deviations it is from the mean along both the x and y axes.
% A simple way to do this is to rescale the axes by the standard
% deviation:

M = [0.5 0 				% matrix with sd along axis
	 0 1.5]

NewDist = inv(M) * Dist;	% rescale distribution using inverse of M
figure(1)
subplot(1, 2, 2)			
plot(NewDist(1, :), NewDist(2, :), '.');
axis([-5 5 -5 5])
axis square
xlabel('amp 1 / sd 1')
ylabel('amp 2 / sd 2')

% now the sd along each axis is 1 and the distribution is symmetrical, and
% moving a distance 1 along either the x or the y axis moves us 1 sd

% Such normalization is useful in a number of cases.  A specific
% neuroscience example is adaptation. Let's say you are interested in how a
% cell responds to 3 different temporal frequencies.  You represent the
% dependence of the cell's responses on frequency in a 3-dimensional space.
% You deliver an adapting stimulus at one frequency.  If this adapting
% stimulus influences the gain of the cell at the same frequency, and only
% this frequency, then you should be able to account for the effects of
% adaptation by scaling the axis corresponding to that frequency. If
% adaptation crosses over to influence gain at the other frequencies, a
% scaling will be needed for those axes as well to account for the effect
% of adaptation.

%%
%-------------------------------------------------------------
% Question set #2:
% (1) Why do rotation matrices take the form above (e.g. the 2-d rotation matrix)?  
%	  Draw a picture and make use of the definitions of sine and cosine from trigonometry.
% (2) What would the matrix look like for a rotation about the x-axis of 30 degrees
%	  followed by a rotation about the z-axis of 45 degrees?
% (3) When a rotation transformation and a scaling transformation are both applied to a
%	  vector, does the order make a difference?  Can you show this?  
% (4) Why do we use the inverse of M above as a scaling matrix?  Why does the inverse
%	  take the form it does?
%-------------------------------------------------------------

%%
%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 4: projection matrices, rank and null spaces
%	
%	Above we considered only invertible matrices.  The columns of
% invertible matrices are independent.  Note that independent does not mean
% orthogonal - it simply means that none of the columns can be created by a
% linear combination of the other columns.  In the standard n equations and
% n unknowns problem, we require that our n equations are independent to
% find a solution - if not the problem is underdetermined.  Similarly, for
% a matrix if we have non-independent columns the inverse is not defined.
% You can see why the inverse is connected to the n equations and n
% unknowns problem by going back to the beginning of the tutorial where we
% used the inverse to solve the 3 equation and 3 unknown problem. The
% reason the inverse is not defined if the columns are not independent is
% because the matrix does not uniquely transform a vector into another
% vector, but instead projects many vectors onto one.  We'll explore this
% in more detail in this section.

%-------------------------------------------------------------
% An example of a noninvertible matrix
%
% A simple example is a matrix that projects all vectors onto the x-y plane
% - in other words, a matrix that eliminates the z component of every
% vector:

M = [1 0 0 
     0 1 0 
	 0 0 0];

% consider the action of this matrix on two different vectors:

v1 = [1
	 0.5
	 1];
v2 = [1
	 0.5
	 0.2];

figure(1)
clf
subplot(1, 2, 1)
Plot3DVector(v1, 'b');
hold on
axis([0 1 0 1 0 1])
Plot3DVector(v2, 'r');
hold off

%%
% now project these vectors through M:

u1 = M * v1;
u2 = M * v2;
	 
figure(1)
subplot(1, 2, 2)
Plot3DVector(u1, 'b');
hold on
axis([0 1 0 1 0 1])
Plot3DVector(u2, 'r');
hold off

% they fall on top of each other - so M has collapsed both vectors onto the
% same line.  Hence there is no way to invert the transformation M made -
% because we can't determine what the value of the z component of the 
% vector should be.

%%
% This property above is not restricted to matrices that eliminate one axis
% of the coordinate system.  For example, the matrix could project all
% vectors to a plane at an angle in the space:

M = [0.5  1    0 
     -0.5 0.5 -0.75 
	 1    0    1];

u1 = M * v1;
u2 = M * v2;
	 
figure(1)
subplot(1, 2, 2)
Plot3DVector(u1, 'b');
hold on
axis([0 1 0 1 0 1])
Plot3DVector(u2, 'r');
hold off

%%
% the projections of v1 and v2 lie on a plane in the space - you can see
% this by rotating the plot:
view(37,56)

% This is not quite in the right plane (so vectors do not obscure each
% other), but you can see that the two vectors lie in a plane.  Try view(34,56) if you
% want to make sure this is true.The section below formalizes these ideas a bit.

%%
%-------------------------------------------------------------
% Null space and column space
%
% A space can be described by a set of axes - e.g. the x-y plane is a space
% spanned by the x and y axis, or equivalently spanned by axes formed by
% x+y and x-y or may other pairs of axes.  The column and null space of
% a matrix determine the set of directions or region of space spanned by the columns in the matrix (the
% column space) and the region that cannot be reached by the columns of the
% matrix (the null space).  A couple general points first.  Each of these
% spaces has an associated dimension - the number of independent axes
% needed to describe the space. The sum of the dimensionality of the column
% space and the null space must be equal to the total number of columns in
% the matrix.

% Although we'll emphasize the column space, matrices also have a row space
% - which is exactly what you would guess - the space spanned by the
% vectors formed from the rows of the matrices.  The number of dimensions
% of the row and column space is equal for all matrices (including non
% square matrices).  

% First the column space.  The number of dimensions of the column space is
% given by the rank of the matrix

M = [0.5  1    0 
     -0.5 0.5 -0.75 
	 1    0    1];
rank(M)

%%
% So we have a column space of rank 2 - i.e. a plane.  If M was an
% invertible 3x3 matrix, it would have rank 3, and if it had only one
% independent column it would have rank 1.  The orth command in Matlab
% provides an orthonormal coordinate system for the column space - i.e. a
% set of orthogonal vectors of unit length.  This is clearer if we think
% about the simple 'project onto the xy plane' matrix:

N = [1 0 0 
     0 1 0 
	 0 0 0];
orth(N)

% note that these two vectors point in the x and y directions and have unit
% length.  This is what we might expect since this matrix projects all
% vectors onto the x-y plane.  This point is made more clearly when we look
% at the null space.  

%%
% The null space consists of the vectors v that satisfy Mv = 0.
% Intuitively, for N above this should be the entire z axis.  Let's check
% that:

null(N)

%%
% Now let's look at the matrix M defined above

orth(M)

% This is another matrix of rank 2, and hence has two vectors in its column
% space.  Let's take a look at these along with our projected vectors (the
% ones we created above where M was defined).  First the original vectors

ColumnSpace = orth(M);
figure(2); clf
Plot3DVector(u1, 'b');
hold on
axis([-1 1 -1 1 -1 1])
Plot3DVector(u2, 'r');

%%
% now the orthonormal coordinate system in black
figure(2)
Plot3DVector(ColumnSpace(:, 1), 'k');
Plot3DVector(ColumnSpace(:, 2), 'k');
hold off
%%

% rotate to see that the black vectors are indeed orthogonal
figure(2)
view(-163,29)

%%
% now rotate to see all these vectors lie in the same plane
figure(2)
view(12,56)

% Ok - so we applied M to two vectors, and said the resulting projections
% lie on a plane.  This plane is the column space of the matrix M.  And the
% orth command gave us a convenient, orthonormal coordinate system for the
% column space.

%%
% The null space for M is a vector perpendicular to the plane.  Let's look
% at that. To make the plot a little simpler, let's just look at the
% orthonormal basis set and the null space: now the orthonormal coordinate
% system in black

figure(2)
clf
hold on
Plot3DVector(ColumnSpace(:, 1), 'k');
Plot3DVector(ColumnSpace(:, 2), 'k');
Plot3DVector(null(M), 'r');
axis([-1 1 -1 1 -1 1])
view(12,56)
hold off

%%
% if you rotate the graph around a bit you can see more clearly the
% relation between these 3 vectors (hold the mouse button down while moving
% the mouse over the graph after executing the line below)

rotate3d on

%%
%-------------------------------------------------------------
% Question set #3:
%	(1) Give an example of a matrix of rank 1.  What are the column space and null spaces for this 
%	    matrix?
%	(2) How many linearly independent vectors are represented in a matrix of rank 5?
%-------------------------------------------------------------

%%
%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 5: Eigensystem and 'natural' coordinates
%
% 	Above we saw that the columns of a vector define a space - the column 
% space.  We also talked about matrices transforming vectors - rotating them, 
% projecting them, etc.  A key concept is the idea that matrices can 
% represent coordinate systems.  Probably the most useful example of this is the 
% eigensystem of a matrix.  That is the topic of this and the next section. We will 
% return to eigensystems in more detail in lecture when we talk about differential 
% equations and again when we discuss dimensional reduction.

% The eigenvalues l and eigenvectors e of a matrix M satisfy
%		M e = l e
% This is important because the eigenvectors are rescaled in length
% but do not change direction when operated on by M.  This means the
% eigenvectors provide a natural coordinate system for M - one in which
% the operation provided by M can be described much more simply than for 
% other choices of coordinates.  Another way of saying this is that M does not
% mix the components or quantities described by the eigenvectors.  Below we'll
% consider how to find the eigenvalues and eigenvectors and illustrate their use with a simple
% mechanical example.  The next section gives another example of 
% their utility.  There are as many eigenvectors as there are dimensions
% in the matrix.  

% You can find eigenvalues analytically from the matrix determinant (it was procedures
% like this and finding the inverse by hand that made me hate linear algebra the first
% time I encountered it!).  For more about this see a linear algebra book
% (e.g. Strang, Linear Algebra and its Applications).  In practice you are more likely
% to find them numerically.  Start with a matrix:

M = [1 -1 0
	 0 3 -1
	 0 0 2];

% the routine eig returns the eigenvalues and eigenvectors.  

[EigVec, EigVal] = eig(M);

% EigVal is a 3x3 diagonal matrix with the eigenvalues along the diagonal. 
% EigVec is a 3x3 matrix whose columns are the eigenvectors.
% Let's check that this works.  First, pull out one eigenvector:

v = EigVec(:, 1)

% and now multiply by M:

M * v

% which is indeed v times the first eigenvalue (which is 1).

%%
%Let's try another

v = EigVec(:, 2)

% and now multiply by M:

M * v
EigVal(2, 2) * v

% try this for some more to make sure you trust Matlab.

%%
%-------------------------------------------------------------
% END OF MAIN TUTORIAL 
%  The section below is optional.  Ignore this section unless you (a) are
%  looking for a way to burn some time; (b) already know something about
%  Markov processes, and (c) are comfortable with the eigensystem stuff
%  above. Otherwise wait on it - we'll come back to these ideas at the end
%  of the course.
%  See Markov model notes on web site for some more information about this
%  section.
%-------------------------------------------------------------

%-------------------------------------------------------------
%-------------------------------------------------------------
% Part 6: Markov chains and state transition matrices
%
%	Markov transition matrices provide a nice example of the utility of
% eigenvectors and eigenvalues. We will see another example in the coming
% weeks when we talk about dimensional reduction.  For the Markov case, the
% eigensystem identifies modes of the system that have characteristic
% kinetics.  This permits us to solve the dynamics of a system of coupled
% states relatively simply.

% We will go through this for the ion channel example.  Our channel has
% three states: closed (C), open (O) and inactivated (I).  The states are
% linked by a set of rate constants.  We'll assume the only permitted
% transitions are:
%		closed <-> open <-> inactive
% Now we will write the probability of being in state x at time t+Dt in
% terms of the probabilities of each state at time t and a transition
% matrix that contains the rate constants (in 1/sec) for transitions from
% one state to the next.  To determine the probability of each state at
% time t we need an initial state, which we will take as all channels
% closed:

InitialS = [1
			0
			0];

% and the transition matrix.  We write this as a Markov transition matrix.
% The entries along the diagonal determine the probability of staying in a
% given state - e.g. the bottom right entry determines the probability of
% staying in the inactivated state at t+Dt if the channel is in the
% inactivated stated at t.  The off diagonal entries determine the
% probabilities of moving from one state to another (as given by the
% transition rate constants and the time step).

M = [0.95  	0.1 	0
	 0.05  	0.75 	0.01
	 0		0.15	0.99];

% this assumes the following:
% time step = 1 msec
% rate constants:
%	closed->open 50/sec
%	open->closed 100/sec
%	open->inactivated 150/sec
%	inactivated->open 10/sec
%   close->inactivated disallowed

% There are two complementary ways to approach the problem of determining
% the evolution of the state probabilities over time.  The first is to use
% the Markov rules (you will see these in the Stochastic Processes
% lectures).  Thus the state vector S at time Dt (one step in) is the
% matrix M applied to the initial state:

S = M * InitialS

% And we can do this recursively over time to get S after N steps (i.e. at time N * Dt):
N = 10;
S = M^N * InitialS;
figure(1)
clf
Plot3DVector(InitialS, 'b');
hold on
Plot3DVector(S, 'r');
xlabel('closed');
ylabel('open');
zlabel('inactivated');

% This allows us to determine the state of the system after any 
% number of steps by a fairly brute force technique.  It has the 
% disadvantage that it does not give us too much intuition about how
% the system is working.  

% Another approach to this same problem is to use the eigensystem. First,
% we rewrite the transition matrix above to separate out the probability
% that we don't change states and to pull out the time step.  We'll define
% a transition matrix

T =	[-50  100    0
	  50 -250   10
	   0  150  -10];

% such that our original Markov matrix is M = I + DeltaT * T where I is the 
% 3x3 identity matrix.  Let's check this:

DeltaT = 0.001;
Identity = [1 0 0 
			0 1 0
			0 0 1];

Identity + DeltaT * T

% Now our update rule takes a form we can interpret using the stuff we learned
% about differential equations and difference equations.  The new state vector S
% at time t+Dt in terms of the state vector at time t is
%	S(t+Dt) = S(t) + Dt T S(t)
% so at time Dt we get

S = InitialS + DeltaT * T * InitialS;

% This should be the same thing we got using M above.  Now, how do we look 
% N steps down the road? This is where we will use the eigenvectors and eigenvalues. 
% First, we can rearrange the update rule for S:
%	[S(t+Dt) - S(t)] / Dt = T S(t)
% The left side is our difference equation version of the derivative, so this
% looks like
%	dS(t)/dt = T S(t)
% and the only complication is that T is a matrix and S a vector!  If S(t) was a
% single variable and T a scalar, we could guess that the solution is something
% like S(t) = A exp(T * t).
% What if S was an eigenvector of T:
%	TS = l S
% where l is the eigenvalue (a scalar).  Then things get much simpler:
%	dS(t)/dt = l S(t)
% but this is just a set of 3 equations (for each component of S), each of 
% which has the simple form 
%	dx(t)/dt = a x(t)
% So the solution in this case (again where S is an eigenvector of T) is
%	S(t) = S(t=0) exp(l t)
% How does this help us?  What we do is express S(t) in terms of the 
% eigenvectors of T:
%	S(t) = sum_n a_n(t) e_n
% where a_n(t) is the amplitude of the eigenvector e_n.  This is just a series
% expansion, like the fourier expansion in sines and cosines, but now the 
% elements of the series are our eigenvectors.
% Then 
%	dS(t)/dt = sum_n e_n da_n(t)/dt
% and 
%	T S(t) = sum_n a_n(t) T e_n = sum_n a_n(t) l_n e_n
% Putting this stuff together we have
% 	sum_n e_n(t) da_n(t)/dt = sum_n a_n(t) l_n e_n
% We can use the orthogonality of the e_n to pull out
% each a_n(t):
%	da_n(t)/dt = l_n a_n(t)
% Hence
%	a_n(t) = a_n(t=0) exp(l_n t)

% Let's work through this in our example above, then come back and interpret
% what is going on.  

[EigVec, EigVal] = eig(T);

% Again EigVec is a 3x3 matrix with columns containing the eigenvectors, and 
% EigVal is a 3x3 diagonal matrix with eigenvalues as the diagonal elements.
% So now we know the l_n (something like -280, -32 and 0).  A couple things.  First, 
% with only 2 nonzero eigenvalues, T has rank 2.  Second, the eigenvalues are negative, 
% which means the associated constants a_n(t) will DECAY exponentially to 0 rather
% than blow up exponentially.  This is a good thing because otherwise we would 
% quickly have problems maintaining the sum of the elements of S being 1 as required.
% This is also why T has to be rank 2 - otherwise ALL the a_n(t) would decay, and we
% would be left with nothing, again running into problems with maintaining the sum
% of elements of S as 1.  

% Our initial S was (1 0 0).  We want to write this as a sum of weighted eigenvectors.
% We can determine the weighting constant using the inverse of the EigVec matrix, whose
% columns are the eigenvectors.  We want:
%	InitialS = EigVec * a
% where a is a vector containing the weights.  So
%	a = inv(EigVec) * InitialS:
% Note that we are using the inverse the same way we used it
% in the 3 equations and 3 unknowns case above, or to generate
% cone isolating stimuli in class.

a = inv(EigVec) * InitialS;

% Let's check this

EigVec * a

% Now we can solve for S at time t:
%	S(t) = sum_n a_n exp(l_n t) e_n
% To compare to above, lets look at S at t=10 msec:

S = 0;
tme = 0.01;
for n = 1:3
	S = S + a(n) * EigVec(:, n) * exp(EigVal(n,n) * tme);
end

S

% we can also immediately solve for the time course of S (e.g. for 1 sec):

clear S;
S(1:1000, 1:3) = 0;
tme = 1:1000;
tme = tme' * 0.001;
for n = 1:3
	for m = 1:3
		S(:, m) = S(:, m) + a(n) .* exp(EigVal(n,n) * tme) .* EigVec(m, n);
	end
end

figure(1)
clf
plot(tme, S);
xlabel('time (sec)')
ylabel('probability')
legend('closed','open','inactivated');

% So what does this tell us?  We have identified characteristic 'modes' of
% the system - in this case combinations of the three states - each of
% which decays with an exponential rate constant. We could track these
% exponential rate constants back to the original transition rate
% constants, and thus get a sense of what determines the behavior.  


% Question set #4 (for your own edification - not to be turned in):
%	(1) Why does the Markov transition matrix have the form it does?
%	(2) What are the steady state probabilities of being in the closed, open
%		and inactivated states?
%	(3) How long does it take the above channel model to get within 1% of the 
%		steady state values?  Can you explain where this time scale comes from?
%	(4) use the orthogonality property of eigenvectors to show the result for 
%		the a_n(t) coefficients above.
%	(5) Confirm that both approaches described above yield the same steady state
%		values for each channel state.


% See also the color space tutorial, the svd tutorial and the 
% regression tutorial.


