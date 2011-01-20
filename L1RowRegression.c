#include <stddef.h>
#include <math.h>
#include "mex.h"

void computeWeights(int nrow, int ncol, double *G, double *z, double *a, double epsilon, double *weights) {
  int i;
  double *zHat = mxCalloc(nrow, sizeof(double));
  char transG = 'N';
  double doubleOne = 1.;
  double doubleZero = 0.;
  int integerOne = 1;

  /* Compute zHat = Ga */
  dgemv_(&transG,  &nrow,  &ncol, &doubleOne, G,   &nrow, a,  &integerOne,
	 &doubleZero, zHat,  &integerOne);
  for (i = nrow; i--;)
    weights[i] = 1./sqrt((zHat[i] - z[i]) * (zHat[i] - z[i]) + epsilon);
  mxFree(zHat);
}

double computeLoss(int nObservations, double *weights) {
  int i;
  double loss = 0.;
  for (i = nObservations; i--; )
    loss += 1./weights[i];
  return loss;
}

void scaleMatrixRowsBySqrtWeights(int nrow, int ncol, double *G, double *weights) {
  int i;
  double w;
  for (i=0; i < nrow; i++) {
    w = sqrt(weights[i]);
    dscal_( &ncol, &w, &G[i],  &nrow);
  }
}

void computeRightHandSide(int nrow, int ncol, double *G, double *weights, double *z, double *zCopy, double *a) {
  int i;
  double *wz;
  char transG = 'T';
  double doubleOne = 1.;
  double doubleZero = 0.;
  int integerOne = 1;
  for (i=nrow; i--; )
    zCopy[i] = weights[i] * z[i];
  dgemv_(&transG,   &nrow,  &ncol, &doubleOne, G,  &nrow, zCopy,  &integerOne, 
	 &doubleZero, a,  &integerOne);
}

void computeLeftHandSide(int nrow, int ncol, double *inputMatrix, double *outputMatrix) {
  char uplo = 'U';
  char transInputMatrix = 'T'; /* Do C:= alpha * A'*A + beta*C */
  double doubleOne = 1.;
  double doubleZero = 0.;
  int integerOne = 1;
  dsyrk_(&uplo, &transInputMatrix,  &ncol,  &nrow, &doubleOne, inputMatrix,  &nrow, 
	 &doubleZero, outputMatrix,  &ncol);
}

void solvePositiveDefinite(int nrow, double *A, double *B) {
  char uplo = 'U';
  int integerOne = 1;
  int INFO;
  dposv_(&uplo,  &nrow,  &integerOne, A,  &nrow, B,  &nrow,  &INFO);
}

//void L1Regression(double *G, double *z, double *a, int nrow, int ncol, int maxiter, double epsilon, double tol, double delta) {  
void L1Regression(double *G, double *z, double *a, int nrow, int ncol, int maxiter, double epsilon, double tol, double mu) {  
  int i, iter;
  double *leftHandSideMatrix, *zCopy, *Gcopy, *weights;
  double loss = 0., lastLoss = 0.;
  
  weights = (double*) mxCalloc(nrow, sizeof(double));
  Gcopy = (double*) mxCalloc(nrow*ncol, sizeof(double));
  zCopy = (double*) mxCalloc(nrow, sizeof(double));
  leftHandSideMatrix = (double*) mxCalloc(ncol*ncol, sizeof(double));
  
  for (iter = maxiter; iter--; ) {

    for (i=ncol*nrow; i--; )
      Gcopy[i] = G[i];

    computeWeights(nrow, ncol, G, z, a, epsilon, weights);
    
    loss = computeLoss(nrow, weights);
    if (fabs(loss - lastLoss) / (fabs(lastLoss) + 1.) < tol)
      break;
    lastLoss = loss;

    /* Compute a = G' * diag(weights) z */
    computeRightHandSide(nrow, ncol, G, weights, z, zCopy, a);

    /* Compute Gcopy = sqrt(diag(weights) * G */
    scaleMatrixRowsBySqrtWeights(nrow, ncol, Gcopy, weights);

    /* Compute leftHandSideMatrix = Gcopy' * Gcopy */
    computeLeftHandSide(nrow, ncol, Gcopy, leftHandSideMatrix);

    /* Add Tychonoff regularization */
    for (i = 0; i < ncol; i++)
      leftHandSideMatrix[i * (ncol + 1)] += mu;

    /* Solve leftHandSideMatrix * x = a; a gets x. */
    solvePositiveDefinite(ncol, leftHandSideMatrix, a);
  }

  /*  printf("Total number of iterations = %d\n", maxiter - iter - 1); */
  mxFree(weights);
  mxFree(Gcopy);
  mxFree(zCopy);
}

void checkForErrors(int nlhs, int nrhs, int nrow, int ncol, int nrowResponse, int ncolResponse, int nrowParameter, int ncolParameter) {
  if (nlhs > 1)
    mexErrMsgTxt("Number of left hand arguments must be 0 or 1.");
  if (nrhs != 7)
    mexErrMsgTxt("Number of right hand arguments must be 7.");
  if (nrow != nrowResponse)
    mexErrMsgTxt("Number of rows of first argument must match number of rows of second argument.");
  if (ncol != nrowParameter)
    mexErrMsgTxt("Number of columns of first argument must match number of rows of third argument.");
  if (ncolResponse != ncolParameter)
    mexErrMsgTxt("Number of columns of second argument must match the number of columns of third argument.");
}

EXTERN_C void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  int nrow, ncol, i, j, iter, maxiter, nL1Regressions;
  double *G, *z, *zCopy, *aInitial, *a;
  double tol = 1e-4, epsilon = 1e-10;
  double mu;
//  double delta = 1e-10;

  /* Error checking to make sure matrices and vectors are all the right size. */
  nrow = mxGetM(prhs[0]);
  ncol = mxGetN(prhs[0]);
  checkForErrors(nlhs, nrhs, nrow, ncol, mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetM(prhs[2]), mxGetN(prhs[2]));

  /* Get arguments from Matlab command line. */
  G = mxGetPr(prhs[0]);                    /* Design matrix is first argument. */
  z = mxGetPr(prhs[1]);                    /* Response vector is second argument. */
  nL1Regressions = (int) mxGetN(prhs[1]);  /* Number of regressions. */
  aInitial = mxGetPr(prhs[2]);             /* Parameter vector is third argument. */
  epsilon = (double) *mxGetPr(prhs[3]);    /* epsilon parameter is fourth argument. */
  maxiter = (int) *mxGetPr(prhs[4]);       /* maxiter parameter is fifth argument. */
  tol = (double) *mxGetPr(prhs[5]);        /* tol parameter is sixth argument. */
//  delta = (double) *mxGetPr(prhs[6]);      /* delta parameter is seventh argument. */
  mu = (double) *mxGetPr(prhs[6]);

  /* Assign output values. */
  plhs[0] = mxCreateDoubleMatrix(ncol, nL1Regressions, mxREAL);
  a = mxGetPr(plhs[0]); 

  /*  mexPrintf("maxiter = %d\n", maxiter); */
  /*  mexPrintf("tol = %g\n", tol); */
  /*  mexPrintf("epsilon = %g\n", epsilon); */
  /*  mexPrintf("delta = %g\n", delta); */

  for (i=0; i < nL1Regressions; i++) {
    for (j=0; j < ncol; j++)
      a[j + i*ncol] = aInitial[j + i*ncol];
//    L1Regression(G, &z[i*nrow], &a[i*ncol], nrow, ncol, maxiter, epsilon, tol, delta);
    L1Regression(G, &z[i*nrow], &a[i*ncol], nrow, ncol, maxiter, epsilon, tol, mu);
  }
}
