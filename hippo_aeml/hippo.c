/* HIPPO i.e.
Haplotype estimation using 
Incomplete 
Prior information from 
Pooled 
Observations

Description:
Hippo is a program for estimating haplotype distribution
from pooled DNA data and is able to utilize prior information
about the haplotypes that are known to be present in the population.
See ? for further details of the model.


   Copyright (C) Matti Pirinen
   
   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   version 2 as published by the Free Software Foundation.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program (gpl.txt); if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   www.gnu.org/licenses/gpl.html
*/


/* 
   Matti Pirinen June-2009: matti.pirinen@iki.fi

   Tested under Linux with gcc version 2.95.4 20011002
   and Gnu Scientific Library version 1.1.1.

   to compile: gcc hippo.c -o hippo -lgsl -lgslcblas -lm

   See hippo_README for further instructions.
*/



#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>



#define ECHO_INPUT 1
#define POISSON_PRIOR 0
#define B_PRIOR 0

struct PARAMETERS_STRUCT{
  int read_haps,variable_list,n_pools,n_loci,n_iterations,n_burnin,write_trace;
  double beta_a,beta_c,beta_mut_1,beta_mut_2,alpha,gamma,proba_mut,tol;
  char data_file[40],hap_file[40];
};


void opening_error(char *str);
void alloc_error(char *str);
int** read_data(FILE *in,int n_loci,int n_pools);
gsl_vector** read_haps(FILE* in,int *n_haps,int n_loci);
double log_2ton_minus_k(double n,double k,double tol);
double loglkhood(gsl_vector **A,gsl_vector * mu,gsl_matrix* u,gsl_vector *eig_val,int *poolsize,int n_pools,int n_loci,int n_eigval,int *not_in_span_u);
gsl_matrix *collect_eigen_directions(gsl_vector* eig_val,gsl_matrix *eig_vec,int n_loci,int *n_eigval);
void print_to_results(FILE *results,gsl_vector **H,int i1,double *psum,double *p2sum);
void print_final_results(FILE *out, FILE *in, int n_loci, double tol);
int search_mutated_hap_from_H(gsl_vector **H,int n_haps,int *SNP_sums,int i1,int l);
double coalescing_probability(gsl_vector **H,double *p,int i1,int i2,int n_haps,int n_loci);
int sample_parent(gsl_vector **H,double *p,int i1,int n_haps,int n_loci,double *sampling_proba,gsl_rng * rng);
void check_rank_of_list(gsl_vector **H,int n_list,int n_loci,gsl_vector **A,int n_pools,double tol);
void complete_list(gsl_vector **H,int *n_haps,int n_loci,gsl_vector **A,int n_pools,double tol);
gsl_vector** all_haps(int *n_haps,int n_loci);
void sample_pair_from_proba(double *p,int n_haps,int *i1,int *i2,gsl_rng * rng);
void read_parameters(FILE *in, struct PARAMETERS_STRUCT *par);

int main(int argc, char* argv[])
{

  /*
    Variables:

    read_haps: if == 1 reads haplotype list from file 'hap_file'
    variable_list: if == 0 list is fixed to the initial one
                      == 1 initial haplotypes are permanent, but additional are allowed
          == 2 no part of the list is fixed, additional are allowed
    n_pools: number of pools
    n_loci: number of loci
    beta_a: additive constant in parameters of beta-proposal distribution
    beta_c: multiplicative constant in parameters of beta-proposal distribution
    beta_mut_1: weight of old haplotype when frequency is divided between old and new haplotypes
    beta_mut_2: weight of new haplotype when frequency is divided between old and new haplotypes
    alpha: prior for haplotypes is Dirichlet(alpha,alpha,...,alpha)
    gamma: penalty for the number of haplotypes (prior is prop to exp(-gamma*n_haps))
    proba_of_proposing_mutation: in RJMCMC algorithm proba of choosing the update that tries to add a new haplotype

    T_lambda: number of haplotypes has Poisson(T_lambda)-prior if POISSON_PRIOR == 1 (not in use currently)
    prior_H, prior_T: if B_PRIOR == 1 then the sum of freqs of known haps has prior beta(prior_H,prior_T) (not in use currently)

    n_list: number of haplotypes in the original list ('m' in the article)
    n_T: number of additional haplotypes, temporary list, ('Lambda^*' in the article) 
    n_H_space: memory has been allocated for n_H_space haplotypes 
    n_haps=n_list+n_T ('nu' in the article)
    poolsize: array of pool sizes in individuals (half of the number of haplotypes in each pool)
    data: temporary storage for data that is read in

    b: sum of the frequencies of haps 0,...,n_list-1 (needed only if B_PRIOR == 1)
    p: freqs of haplotypes (theta in the article)
    psum: sum of the freqs of haps over iterations, reset if hap is deleted
    p2sum: sum of the squared freqs of haps over iterations, reset if hap is deleted
   */

  struct PARAMETERS_STRUCT par;

  int n_pools, n_loci;
  double beta_a,beta_c,beta_mut_1,beta_mut_2,alpha,gamma,proba_of_proposing_mutation;
  double T_lambda=10,prior_H=8,prior_T=2; //only for B_PRIOR and POISSON_PRIOR, not in use currently
 
  int n_list,n_T,n_H_space,n_haps,*poolsize,**data;
  //counters:
  int accept_mutations=0,accept_deletions=0,accept=0,accept_substitutions=0,not_in_span_u=0,not_in_span_u2=0;
  int reject_mutation=0,reject_deletion=0,reject=0,reject_substitutions=0,move_away=0,move_towards=0,accepted_iterations=0;
  
  int i,j,i1,i2,iter,n_eigval,l,random_pair,try_to_mutate,*SNP_sums;;
  FILE *in,*monitor,*results,*trace;
  gsl_vector **A,**H,*mu,*eig_val;
  gsl_matrix *sigma,*eig_vec,*u;
  double *p,*p2,*psum,*p2sum,logl,logl2,logposterior,logposterior_max,logHR,b,b2,bsum=0.0,c,sum,sampling_proba,v,v2;
  void *temp_pointer;
  gsl_eigen_symmv_workspace *eig_work;
  gsl_rng * rng = gsl_rng_alloc (gsl_rng_taus);

  if(argc!=2)
    {fprintf(stderr,"Usage:./hippo 'parameters_file' (see README)\n");exit(1);}

  if((in=fopen(argv[1],"r"))==NULL)
    {
      fprintf(stderr,"Cannot open file %s\n",argv[1]);
      exit(1);
    }

  printf("\n*******************************************************************************************\n");
  printf("* HIPPO (Haplotype estimation under Incomplete Prior information from Pooled Observations *\n");
  printf("*******************************************************************************************\n"); 
  printf("\nReading parameters from file '%s'\n\n",argv[1]);
  read_parameters(in,&par);
  fclose(in);

  n_pools=par.n_pools; n_loci=par.n_loci;
  beta_a=par.beta_a;beta_c=par.beta_c;beta_mut_1=par.beta_mut_1;beta_mut_2=par.beta_mut_2;
  alpha=par.alpha;gamma=par.gamma;proba_of_proposing_mutation=par.proba_mut;

  if((in=fopen("seed","r"))==NULL)
    gsl_rng_set (rng, time(NULL)); //seed
  else
    {fscanf(in,"%d",&i);gsl_rng_set (rng,i);printf("Using seed %d\n",i);fclose(in);} 
  
  if(par.read_haps == 1)
    {
      if((in=fopen(par.hap_file,"r"))==NULL)
  opening_error(par.hap_file);
      H=read_haps(in,&n_list,n_loci);
      printf("\n");
      fclose(in);
    } 
  else
    {
      if(n_loci>10){ //upper bound may be changed here
  fprintf(stderr,"\nNumber of Loci=%d > 10, results in too many haplotypes! \n",n_loci);
  fprintf(stderr,"Decrease the number of loci or specify a prior list of haplotypes.\n\n");
  exit(2);
      }
      H=all_haps(&n_haps,n_loci);
      n_list=n_haps;
    }
  
  if((in=fopen(par.data_file,"r"))==NULL)
    opening_error(par.data_file);
  data=read_data(in,n_loci,n_pools);
  fclose(in);
 
  n_T=0;
  n_haps=n_list;
  n_H_space=n_list+n_loci;

  if((H=(gsl_vector**) realloc(H,(n_H_space)*sizeof(gsl_vector*)))==NULL)
    alloc_error("H");
  for(i=n_list;i<n_H_space;++i)
    H[i]=gsl_vector_alloc(n_loci);

  if((p=(double*) malloc(n_H_space*sizeof(double)))==NULL)
    alloc_error("p");
  if((p2=(double*) malloc(n_H_space*sizeof(double)))==NULL)
    alloc_error("p2");
  if((psum=(double*) calloc(n_H_space,sizeof(double)))==NULL)
    alloc_error("psum");
  if((p2sum=(double*) calloc(n_H_space,sizeof(double)))==NULL)
    alloc_error("p2sum");
  if((SNP_sums=(int*) calloc(n_H_space,sizeof(int)))==NULL)
    alloc_error("SNP_sums");

  if((poolsize=(int*)malloc(n_pools*sizeof(int)))==NULL)
    alloc_error("poolsize");
  
  if((A=(gsl_vector**) malloc(n_pools*sizeof(gsl_vector*)))==NULL)
    alloc_error("A");

  for(i=0;i<n_pools;++i)
    {
      poolsize[i]=data[i][0];
      A[i]=gsl_vector_alloc(n_loci);
      for(j=0;j<n_loci;++j)
  gsl_vector_set(A[i],j,data[i][j+1]);
      free(data[i]);
    }
  free(data);

  //check_rank_of_list(H,n_haps,n_loci,A,n_pools,1e-3);
  complete_list(H,&n_haps,n_loci,A,n_pools,1e-3); //if data is not is span(H^t) then H is complemented
  n_T=n_haps-n_list;

  //allocate variables
  mu=gsl_vector_calloc(n_loci);
  sigma=gsl_matrix_calloc(n_loci,n_loci);
  eig_val=gsl_vector_calloc(n_loci);
  eig_vec=gsl_matrix_calloc(n_loci,n_loci);
  eig_work=gsl_eigen_symmv_alloc(n_loci);

  for(i=0;i<n_haps;++i)
    for(j=0;j<n_loci;++j)
      SNP_sums[i]+=gsl_vector_get(H[i],j);

  //initialize variables p,mu,sigma  
  for(i=0;i<n_haps;++i)
    {
      p[i]=1.0/n_haps;
      gsl_blas_daxpy(p[i],H[i],mu); //mu+=p[i]*H[i]
      gsl_blas_dsyr(CblasLower, p[i], H[i], sigma); //sigma+=p[i]*H[i]*H[i]^T
    }
  b=1.0*n_list/n_haps;
  
  gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T  
  gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
  u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval);
  if(n_eigval==0){fprintf(stderr,"Initial covariance matrix is zero matrix!");exit(5);}
  logl=loglkhood(A,mu,u,eig_val,poolsize,n_pools,n_loci,n_eigval,&not_in_span_u);      
  gsl_matrix_free(u);

  logposterior_max=-1.0/0.0;
  monitor=fopen("monitor.out","w");
  results=fopen("results.raw","w");
  if(par.write_trace) trace=fopen("trace.out","w");

  for(i=0;i<n_haps;++i) p2[i]=p[i];

  printf("\n\nStarting MCMC\n");

  for(iter=0;iter<par.n_iterations;++iter)
    {
      if(iter%10000==9999) {
  printf("%d iterations, n_haps=%d, n_T=%d, logl=%g, dim=%d\n",iter+1,n_haps,n_T,logl,n_eigval);
  printf(" acc/rej=%d/%d, acc/rej_mutations=%d/%d, \n acc/rej_deletions=%d/%d acc/rej_substitutions=%d/%d \n",
         accept,reject,accept_mutations,reject_mutation,accept_deletions,reject_deletion,accept_substitutions,reject_substitutions);
      }
      //M-H STEP1:UPDATING FREQUENCIES OF CURRENT HAPLOTYPES

      random_pair=(iter%3);//(gsl_rng_uniform(rng)<0.5);

      if(random_pair)
  {
    i1=gsl_rng_get(rng)%n_haps;
    i2=gsl_rng_get(rng)%(n_haps-1);
    if(i2>=i1) i2+=1;
    else{i=i2;i2=i1;i1=i;} //now i2>i1
  }
      else
  sample_pair_from_proba(p,n_haps,&i1,&i2,rng);

      //p2=p at all indexes at this point
      c=gsl_ran_beta(rng,beta_a+p[i1]*beta_c,beta_a+p[i2]*beta_c);
      if(c>0.99999) c=0.99999;
      if(c<0.00001) c=0.00001;
      p2[i1]=(p[i1]+p[i2])*c;
      p2[i2]=(p[i1]+p[i2])-p2[i1];
      if(gsl_isnan(p2[i1]))
  {printf("NAN in beta generation\nIncrease parameter beta_a.");exit(1);}

      gsl_vector_set_zero(mu);    
      gsl_matrix_set_zero(sigma);
      for(i=0;i<n_haps;++i)
  {
    gsl_blas_daxpy(p2[i],H[i],mu); //mu+=p2[i]*H[i]
    gsl_blas_dsyr(CblasLower, p2[i], H[i], sigma); //sigma+=p2[i]*H[i]*H[i]^T
  }
      gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T
      gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
      u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval); 
      if(n_eigval>0){
  logl2=loglkhood(A,mu,u,eig_val,poolsize,n_pools,n_loci,n_eigval,&not_in_span_u2);      
  gsl_matrix_free(u);
    
  logHR=logl2-logl; 
  logHR+=log(gsl_ran_beta_pdf(p[i1]/(p[i1]+p[i2]),beta_a+p2[i1]*beta_c,beta_a+p2[i2]*beta_c))
    -log(gsl_ran_beta_pdf(p2[i1]/(p[i1]+p[i2]),beta_a+p[i1]*beta_c,beta_a+p[i2]*beta_c));
  logHR+=(alpha-1)*(log(p2[i1])+log(p2[i2])-log(p[i1])-log(p[i2]));
  
  if(!random_pair)
    logHR+=log(p2[i1])+log(p2[i2])+log(1-p[i1])+log(1-p[i2])-(log(p[i1])+log(p[i2])+log(1-p2[i1])+log(1-p2[i2]));

  if(par.variable_list < 2) 
    {
      if(i1<n_list && i2>=n_list)
        {
    b2=b-p[i1]+p2[i1];
#if B_PRIOR == 1
    logHR+=(prior_H-1)*(log(b2)-log(b))+(prior_T-1)*(log(1-b2)-log(1-b));
#endif
        }
      else
        b2=b;
    }
      }

      if(gsl_isnan(logHR))
  {fprintf(stderr,"NaN in logHR %g %g %g %g\n",p[i1],p[i2],p2[i1],p2[i2]);exit(1);}

      if(n_eigval>0 && log(gsl_rng_uniform(rng))<=logHR && not_in_span_u2<=not_in_span_u)
  {//proposal accepted
    p[i1]=p2[i1];
    p[i2]=p2[i2];
    logl=logl2;
    b=b2;
    move_away+=(not_in_span_u2>not_in_span_u);
    move_towards+=(not_in_span_u2<not_in_span_u);
    not_in_span_u=not_in_span_u2;
    ++accept;
  }
      else
  {//proposal rejected
    ++reject;
    p2[i1]=p[i1];
    p2[i2]=p[i2];
  }



      if(par.variable_list >= 1)
  {

      if(iter%1==0)
  {
    //RJMCMC STEP: PROPOSE A MUTATION TO AN EXISTING HAPLOTYPE
    i1=gsl_rng_get(rng)%n_haps;
    l=gsl_rng_get(rng)%n_loci;
    if(par.variable_list < 2)
      {
        if(i1<n_list) try_to_mutate=1;
        else try_to_mutate=(gsl_rng_uniform(rng)<proba_of_proposing_mutation);
      }    
    else
      {
        if(n_haps<=2) try_to_mutate=1; //take this possibility also into account in HR!!!!!
        else try_to_mutate=(gsl_rng_uniform(rng)<proba_of_proposing_mutation);
      }

    if(try_to_mutate)
      {
        i2=search_mutated_hap_from_H(H,n_haps,SNP_sums,i1,l);
        if(i2>-1)
    {//mutated haplotype is already in H, do nothing
    }
        else
    {//mutated haplotype is not in H, create it and give it some frequency from i1
      if(n_H_space<=n_haps)
        {
          n_H_space+=1+n_loci;
          if((H=(gsl_vector**) realloc(H,(n_H_space)*sizeof(gsl_vector*)))==NULL)
      alloc_error("H");
          for(i=n_haps;i<n_H_space;++i)
      H[i]=gsl_vector_alloc(n_loci);
          if((p=(double*) realloc(p,n_H_space*sizeof(double)))==NULL)
      alloc_error("p");
          if((p2=(double*) realloc(p2,n_H_space*sizeof(double)))==NULL)
      alloc_error("p2");
          if((psum=(double*) realloc(psum,n_H_space*sizeof(double)))==NULL)
      alloc_error("psum");
          if((p2sum=(double*) realloc(p2sum,n_H_space*sizeof(double)))==NULL)
      alloc_error("p2sum");
          if((SNP_sums=(int*) realloc(SNP_sums,n_H_space*sizeof(int)))==NULL)
      alloc_error("SNP_sums");
        }

      for(i=0;i<n_loci;++i)
        gsl_vector_set(H[n_haps],i,gsl_vector_get(H[i1],i));
      gsl_vector_set(H[n_haps],l,1-gsl_vector_get(H[i1],l));
        
      c=gsl_ran_beta(rng,beta_mut_1,beta_mut_2);
      p2[i1]=p[i1]*c;
      p2[n_haps]=p[i1]-p2[i1];

      if(par.variable_list < 2)
        {
          if(i1<n_list)
      b2=b-p2[n_haps];
          else
      b2=b;
        }

      gsl_vector_set_zero(mu);    
      gsl_matrix_set_zero(sigma);
      for(i=0;i<=n_haps;++i) //also the new hap is included!
        {
          gsl_blas_daxpy(p2[i],H[i],mu); //mu+=p2[i]*H[i]
          gsl_blas_dsyr(CblasLower, p2[i], H[i], sigma); //sigma+=p2[i]*H[i]*H[i]^T
        }
      gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T
      gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
      u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval);  
      if(n_eigval>0){
        logl2=loglkhood(A,mu,u,eig_val,poolsize,n_pools,n_loci,n_eigval,&not_in_span_u2);      
        gsl_matrix_free(u);
        
        //computing log(HR)
        logHR=logl2-logl; //likelihood

#if POISSON_PRIOR == 1 
        logHR+=log(T_lambda)-log_2ton_minus_k(n_loci,n_haps,0.001); //priors
#endif
#if B_PRIOR == 1
        if(fabs(b-1.0)<0.00001) 
          b=0.99999;
        logHR+=(prior_H-n_list-1)*(log(b2)-log(b))+(prior_T-n_T-1)*(log(1-b2)-log(1-b))-log(1-b2); //priors
#endif

        logHR-=gamma;
        //logHR+=gsl_sf_lngamma(alpha*(n_haps+1))-gsl_sf_lngamma(alpha*n_haps)-gsl_sf_lngamma(alpha);//prior for p
        logHR+=(alpha-1)*(log(p2[i1])+log(p2[n_haps])-log(p[i1])); //prior for p

        //logHR+=-log((n_haps+1)*n_loci)+log(1-proba_of_proposing_mutation);//inverse transition probability, old version
        logHR+=-log(n_haps+1)+log(1-proba_of_proposing_mutation)+log(coalescing_probability(H,p2,n_haps,i1,n_haps+1,n_loci));//inverse transition probability
        logHR+=log(n_haps*n_loci)-log(gsl_ran_beta_pdf (c,beta_mut_1,beta_mut_2)); //transition probability

        if(par.variable_list < 2 && i1>=n_list) 
          logHR-=log(proba_of_proposing_mutation); //transition proba includes proba of mutation 
        logHR+=log(p[i1]); //|Jacobian|

        if(gsl_isnan(logHR))
          {printf("NAN in mutation HR calculation\n");exit(1);}
      }

      if(n_eigval>0 && log(gsl_rng_uniform(rng))<=logHR && not_in_span_u2<=not_in_span_u)
        {//proposal accepted
          //printf("ACCEPT MUTATION\n");
          ++n_haps;
          ++n_T;
          p[i1]=p2[i1];
          p[n_haps-1]=p2[n_haps-1];
          psum[n_haps-1]=0.0;
          p2sum[n_haps-1]=0.0;
          logl=logl2;
          b=b2;
          move_away+=(not_in_span_u2>not_in_span_u);
          move_towards+=(not_in_span_u2<not_in_span_u);
          not_in_span_u=not_in_span_u2;
          ++accept_mutations;
          SNP_sums[n_haps-1]=SNP_sums[i1];
          SNP_sums[n_haps-1]+=-1+2*gsl_vector_get(H[n_haps-1],l);
        }
      else
        {//proposal rejected
          //printf("REJECT MUTATION\n");
          ++reject_mutation;
          p2[i1]=p[i1];
          p2[n_haps]=0.0;
          b2=b;
        }
    }
      }
    else
      {//try to delete the haplotype i1 by coalesing it to some hap i2 that is one mutation away from i1
        i2=sample_parent(H,p,i1,n_haps,n_loci,&sampling_proba,rng);
        if(i2==-1)
    {//no possible parents currently in H, do nothing
    }
        else
    {
      //printf("Trying to delete %d by coalescing to %d\n",i1,i2);
      p2[i2]=p[i2]+p[i1];
      p2[i1]=0.0;
      c=p[i2]/p2[i2];

      if(par.variable_list < 2)
        if(i2<n_list)
          b2=b+p[i1];
        else
          b2=b;

      //computing likelihood of the proposal 
      //should i1 be left out from calculations or is it enough that p2[i1]=0.0?
      gsl_vector_set_zero(mu);    
      gsl_matrix_set_zero(sigma);
      for(i=0;i<n_haps;++i) 
        {
          gsl_blas_daxpy(p2[i],H[i],mu); //mu+=p2[i]*H[i]
          gsl_blas_dsyr(CblasLower, p2[i], H[i], sigma); //sigma+=p2[i]*H[i]*H[i]^T
        }
      gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T
      gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
      u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval);  
      if(n_eigval>0){
        logl2=loglkhood(A,mu,u,eig_val,poolsize,n_pools,n_loci,n_eigval,&not_in_span_u2);      
        gsl_matrix_free(u);
        
        //computing log(HR)
        logHR=logl2-logl; //likelihood

#if POISSON_PRIOR == 1 
        logHR+=-log(T_lambda)+log_2ton_minus_k(n_loci,n_haps-1,0.001); //priors
#endif
#if B_PRIOR == 1
        if(fabs(b2-1.0)<0.00001) b2=0.99999;//in order to not take log(0.0)
        logHR+=(prior_H-n_list-1)*(log(b2)-log(b))+(prior_T-n_T)*(log(1-b2)-log(1-b))+log(1-b); //priors
#endif
        logHR+=gamma;
        //logHR+=-gsl_sf_lngamma(alpha*(n_haps+1))+gsl_sf_lngamma(alpha*n_haps)+gsl_sf_lngamma(alpha);//prior for p
        logHR+=(alpha-1)*(log(p2[i2])-log(p[i1])-log(p[i2])); //prior for p

        //logHR-=-log((n_haps)*n_loci)+log(1-proba_of_proposing_mutation);//transition probability, old version
        logHR-=-log(n_haps)+log(1-proba_of_proposing_mutation)+log(sampling_proba);//transition probability
        logHR+=-log((n_haps-1)*n_loci)+log(gsl_ran_beta_pdf (c,beta_mut_1,beta_mut_2)); //inverse transition probability

        if(par.variable_list < 2 && i2>=n_list) 
          logHR+=log(proba_of_proposing_mutation); //inverse transition proba includes proba of mutation 
        logHR+=-log(p[i1]+p[i2]); //|Jacobian|

        if(gsl_isnan(logHR))
          {printf("NAN in deletion HR calculation\n");exit(1);}
      }

      if(n_eigval>0 && log(gsl_rng_uniform(rng))<=logHR && not_in_span_u2<=not_in_span_u)
        {//proposal accepted
          if(iter>=par.n_burnin)
      {
        // fprintf(results, "%d\n", iter);
        print_to_results(results,H,i1,psum,p2sum);
      }
          p[i2]=p2[i2];
          p[i1]=p2[i1];
          b=b2;
          temp_pointer=H[i1];
          H[i1]=H[n_haps-1];
          H[n_haps-1]=(gsl_vector*) temp_pointer;
          SNP_sums[i1]=SNP_sums[n_haps-1];
          SNP_sums[n_haps-1]=0;
          psum[i1]=psum[n_haps-1];
          psum[n_haps-1]=0.0;
          p2sum[i1]=p2sum[n_haps-1];
          p2sum[n_haps-1]=0.0;
          p[i1]=p[n_haps-1];
          p[n_haps-1]=0.0;
          p2[i1]=p[i1];
          p2[n_haps-1]=0.0;
          --n_haps;
          --n_T;
          logl=logl2;
          ++accept_deletions;
          move_away+=(not_in_span_u2>not_in_span_u);
          move_towards+=(not_in_span_u2<not_in_span_u);
          not_in_span_u=not_in_span_u2;
        }
      else
        {//proposal rejected
          ++reject_deletion;
          p2[i2]=p[i2];
          p2[i1]=p[i1];
          b2=b;
        }      
    }
      }
  }

      //M-H STEP3:TRYING TO SUBSTITUTE A CURRENT HAPLOTYPE IN T BY ANOTHER ONE NOT IN H
      if(par.variable_list < 2)
  {
    if(n_T>0)
      i1=n_list+gsl_rng_get(rng)%n_T;
  }
      else
  {
    if(n_haps>0)
      i1=gsl_rng_get(rng)%n_haps;
  }
      l=gsl_rng_get(rng)%n_loci;
      i2=search_mutated_hap_from_H(H,n_haps,SNP_sums,i1,l);
    
      if( (par.variable_list < 2 && (n_T==0 || i2>-1)) || (par.variable_list > 0 && (n_haps==0 || i2>-1))  )
  {//no haps to substitute or mutated haplotype is already in H, do nothing
  }
      else
  {
    gsl_vector_set(H[i1],l,1-gsl_vector_get(H[i1],l));//change i1 at l 
    gsl_vector_set_zero(mu);    
    gsl_matrix_set_zero(sigma);
    for(i=0;i<n_haps;++i) 
      {
        gsl_blas_daxpy(p[i],H[i],mu); //mu+=p[i]*H[i]
        gsl_blas_dsyr(CblasLower, p[i], H[i], sigma); //sigma+=p[i]*H[i]*H[i]^T
      }
    gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T
    gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
    u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval);  
    if(n_eigval>0){
      logl2=loglkhood(A,mu,u,eig_val,poolsize,n_pools,n_loci,n_eigval,&not_in_span_u2);      
      gsl_matrix_free(u);
      
      //computing log(HR)
      logHR=logl2-logl; //likelihood is the only contribution here!
    }
    if(gsl_isnan(logHR))
      {printf("NAN in substitute HR calculation\n");exit(1);}
    if(n_eigval>0 && log(gsl_rng_uniform(rng))<=logHR && not_in_span_u2<=not_in_span_u)
      {//proposal accepted
        ++accept_substitutions;
        SNP_sums[i1]+=-1+2*gsl_vector_get(H[i1],l);
        logl=logl2;
        move_away+=(not_in_span_u2>not_in_span_u);
        move_towards+=(not_in_span_u2<not_in_span_u);
        not_in_span_u=not_in_span_u2;
      }
    else
      {//proposal rejected
        ++reject_substitutions;
        gsl_vector_set(H[i1],l,1-gsl_vector_get(H[i1],l));//change i1 back to its original form
      }
  }
  } //if variable_list >= 0

      sum=0.0;
      for(i=0;i<n_haps;++i)
  sum+=log(p[i]);
      sum*=(alpha-1);
      logposterior=logl+sum-n_haps*gamma;

      if(iter%100==0)
  fprintf(monitor,"%10.5g %10.5g %10.5g %10.5g\n",logposterior,logl,sum,-n_haps*gamma);
      //prints: logposterior, loglikelihood of data, prior for p, prior for n_haps

      if(not_in_span_u)
  {
    fprintf(stderr,"NOT in span U\n");exit(1);
  }
      if(iter>=par.n_burnin && !not_in_span_u)
  {
    for(i=0;i<n_haps;++i)
      {
        psum[i]+=p[i];
        p2sum[i]+=p[i]*p[i];
      }
    bsum+=b;
    ++accepted_iterations;
  }

      if(par.variable_list < 2)
  {
    sum=0.0;
    for(i=0;i<n_list;++i)
      sum+=p[i];
    if(fabs(b-sum)>0.001)
      {fprintf(stderr,"ERROR b!=sum_H p, %g %g\n",b,sum);exit(8);}
      
    for(i=n_list;i<n_haps;++i)
      sum+=p[i];
    if(fabs(1.0-sum)>0.001)
      {
        printf("%d %g\n",iter,b);
        fprintf(stderr,"ERROR sum_haps p=%g != 1.0\n",sum);exit(8);
      }
  }

  if(par.write_trace && iter>=par.n_burnin) {
    fprintf(trace,"%d\n", n_haps);
  for(i=0;i<n_haps;++i)
    {
      for(j=0;j<n_loci;++j)
  fprintf(trace,"%g",gsl_vector_get(H[i],j));
      fprintf(trace," %g\n",p[i]);
    }
  }

      //print out the MAP estimate
      if(!not_in_span_u && logposterior>logposterior_max)
  {
    FILE *out=fopen("MAP.out","w");
    fprintf(out," %g\n",logposterior);
    for(i=0;i<n_haps;++i)
      {
        for(j=0;j<n_loci;++j)
    fprintf(out,"%g",gsl_vector_get(H[i],j));
        fprintf(out," %g\n",p[i]);
      }
    fclose(out);
    logposterior_max=logposterior;
  }
    }

  fclose(monitor);

  printf("Acceptance ratios:%g %g %g %g\n",
   accept/1.0/par.n_iterations,accept_mutations/1.0/par.n_iterations,accept_deletions/1.0/par.n_iterations,accept_substitutions/1.0/par.n_iterations);
  printf("accepted_iterations=%d, move away=%d, move in=%d\n",accepted_iterations,move_away,move_towards);
 #if B_PRIOR == 1
  printf("E(b)=%g\n",bsum/(par.n_iterations-par.n_burnin));
#endif
  for(i=0;i<n_haps;++i)
    print_to_results(results,H,i,psum,p2sum);

  //fprintf(results,"%d %d\n",par.n_iterations,par.n_burnin);
  fclose(results);

  if((in=fopen("results.raw","r"))==NULL)
    opening_error("results.raw");
  results=fopen("results.out","w");
  print_final_results(results,in,n_loci,par.tol);
  fclose(results);
  fclose(in);

  {
    
    FILE *out; 
    /*
    out=fopen("acceptance.out","a");
    fprintf(out,"Acceptance ratios:%g %g %g %g ",
      accept/1.0/par.n_iterations,accept_mutations/1.0/par.n_iterations,accept_deletions/1.0/par.n_iterations,accept_substitutions/1.0/par.n_iterations);
    fprintf(out,"%d %d\n",move_away,move_towards);
    fclose(out);
    */
    
    out=fopen("seed","w");
    fprintf(out,"%ld\n",gsl_rng_get(rng));
    fclose(out);
  }
  
  free(poolsize);
  gsl_rng_free(rng);

  return 0;
}

#include "hippo_functions.c"
