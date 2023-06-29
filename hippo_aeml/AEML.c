/* AEML i.e.
Approximative 
EM -algorithm that uses 
List of known haplotypes.


Description:
AEML is a program for estimating haplotype distribution
from pooled DNA data and is able to utilize prior information
about the haplotypes that are known to be present in the population.
AEML is an extension of AEM:
Kuk et al. Computationally feasible estimation of haplotype frequencies 
from pooled DNA with and without Hardy–Weinberg equilibrium
Bioinformatics 2009:25(3):379-386.

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

   to compile: gcc AEML.c -o AEML -lgsl -lgslcblas -lm

   See AEML_README for further instructions.
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


// gcc AEML.c -o AEML -lgsl -lgslcblas -lm


#define ECHO_INPUT 1


struct PARAMETERS_STRUCT{
  int read_haps,n_pools,n_loci,n_iterations,random_init;
  double tol,stab;
  char data_file[40],hap_file[40];
};


void opening_error(char *str);
void alloc_error(char *str);
int** read_data(FILE *in,int n_loci,int n_pools);
gsl_vector** read_haps(FILE* in,int *n_haps,int n_loci);
gsl_matrix *collect_eigen_directions(gsl_vector* eig_val,gsl_matrix *eig_vec,int n_loci,int *n_eigval);
void check_rank_of_list(gsl_vector **H,int n_list,int n_loci,gsl_vector **A,int n_pools,double tol);
gsl_vector** all_haps(int *n_haps,int n_loci);
void read_parameters(FILE *in, struct PARAMETERS_STRUCT *par);

int main(int argc, char* argv[])
{

  struct PARAMETERS_STRUCT par;
  int n_pools,n_loci,n_haps,*poolsize,**data,i,j,h,iter,n_eigval,not,n_ind,go_on;
  FILE *in,*monitor,*results;
  gsl_vector **A,**H,*mu,*eig_val,*y,*yy,*y_minus_mu,*x;
  gsl_matrix *sigma,*eig_vec,*u;
  double *p,*expected,sum,logl,logl2,val,val2,logdet,v;
  gsl_eigen_symmv_workspace *eig_work;
  gsl_rng * rng = gsl_rng_alloc (gsl_rng_taus);

  if(argc!=2)
    {fprintf(stderr,"Usage:./AEML 'parameters_file' (see AEML_README)\n");exit(1);}

  if((in=fopen(argv[1],"r"))==NULL)
    {
      fprintf(stderr,"Cannot open file %s\n",argv[1]);
      exit(1);
    }

  printf("\n******************************************************************\n");
  printf("* AEML (Approximate EM-algorithm using List of known haplotypes) *\n");
  printf("******************************************************************\n"); 
  printf("\nReading parameters from file '%s'\n\n",argv[1]);
  read_parameters(in,&par);
  fclose(in);

  n_pools=par.n_pools; n_loci=par.n_loci;

  if(par.random_init==1)
    {
      if((in=fopen("AEML_seed","r"))==NULL)
	gsl_rng_set (rng, time(NULL)); //seed
      else
	{fscanf(in,"%d",&i);gsl_rng_set (rng,i);printf("Using seed %d\n",i);} 
      fclose(in);
    }

  if(par.read_haps == 1)
    {
      if((in=fopen(par.hap_file,"r"))==NULL)
	opening_error(par.hap_file);
      H=read_haps(in,&n_haps,n_loci);
      printf("\n");
      fclose(in);
    } 
  else
    {
      if(n_loci>10){
	fprintf(stderr,"\nNumber of Loci=%d > 10, results in too many haplotypes! \n",n_loci);
	fprintf(stderr,"Decrease the number of loci or specify a prior list of haplotypes.\n\n");
	exit(2);
      }
      H=all_haps(&n_haps,n_loci);
    }

  if((in=fopen(par.data_file,"r"))==NULL)
    opening_error(par.data_file);
  data=read_data(in,n_loci,n_pools);
  fclose(in);
 

  if((p=(double*) malloc(n_haps*sizeof(double)))==NULL)
    alloc_error("p");

  if((expected=(double*) malloc(n_haps*sizeof(double)))==NULL)
    alloc_error("p");
 
  if((poolsize=(int*)malloc(n_pools*sizeof(int)))==NULL)
    alloc_error("poolsize");
  
  if((A=(gsl_vector**) malloc(n_pools*sizeof(gsl_vector*)))==NULL)
    alloc_error("A");

  n_ind=0;
  for(i=0;i<n_pools;++i)
    {
      poolsize[i]=data[i][0];
      n_ind+=poolsize[i];
      A[i]=gsl_vector_alloc(n_loci);
      for(j=0;j<n_loci;++j)
	gsl_vector_set(A[i],j,data[i][j+1]);
      free(data[i]);
    }
  free(data);

  check_rank_of_list(H,n_haps,n_loci,A,n_pools,1e-3);

  mu=gsl_vector_calloc(n_loci);
  sigma=gsl_matrix_calloc(n_loci,n_loci);
  eig_val=gsl_vector_calloc(n_loci);
  eig_vec=gsl_matrix_calloc(n_loci,n_loci);
  eig_work=gsl_eigen_symmv_alloc(n_loci);
  y=gsl_vector_alloc(n_loci);
  y_minus_mu=gsl_vector_alloc(n_loci);

  if(par.random_init==1)
    {
      sum=0.0;
      for(i=0;i<n_haps;++i)
	{
	  p[i]=0.5+gsl_rng_uniform(rng);
	  sum+=p[i];
	}
      for(i=0;i<n_haps;++i)
	p[i]=p[i]/sum;
      sum=0.0;
    }
  else
    for(i=0;i<n_haps;++i)
      p[i]=1.0/n_haps; //initializing
  
  monitor=fopen("AEML_monitor.out","w");
  go_on=1;iter=0;logl=-1.0/0.0;

  while(go_on && iter<par.n_iterations)
    {  
      gsl_vector_set_zero(mu);	  
      gsl_matrix_set_zero(sigma);
      for(i=0;i<n_loci;++i) gsl_matrix_set(sigma, i, i, par.stab);
      for(i=0;i<n_haps;++i)
	{
    expected[i]=0.0;
	  gsl_blas_daxpy(p[i],H[i],mu); //mu+=p[i]*H[i]
	  gsl_blas_dsyr(CblasLower, p[i], H[i], sigma); //sigma+=p[i]*H[i]*H[i]^T
	}
      gsl_blas_dsyr(CblasLower,-1.0,mu,sigma);//sigma-=mu*mu^T  
      gsl_eigen_symmv(sigma, eig_val,eig_vec,eig_work);
      u=collect_eigen_directions(eig_val,eig_vec,n_loci,&n_eigval);
      yy=gsl_vector_alloc(n_eigval);
      x=gsl_vector_alloc(n_eigval);

      logdet=0.0;
      for(i=0;i<n_eigval;++i)
	logdet+=log(gsl_vector_get(eig_val,i));

      logl2=logl; //logl2 is now from the previous iteration
      logl=0.0;
      sum=0.0;

      for(i=0;i<n_pools;++i)
	{

	  gsl_vector_memcpy (y_minus_mu,A[i]); //y_minus_mu=A[i]
	  gsl_blas_daxpy(-poolsize[i],mu,y_minus_mu); //y=y_minus_mu-poolsize*mu
	  gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0,yy);//yy=u y

	  gsl_blas_dgemv (CblasTrans, 1.0, u,yy,0.0,y);//y=u^t yy
	  gsl_blas_daxpy (-1.0, y_minus_mu, y);
	  val=gsl_blas_dnrm2(y);
	  if(val>0.001)
	    {not=1;}//not is 1 if A[i]-mu is not in the space spanned by columns of u
	  
	  //gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0, yy); //yy=1.0*u*y_minus_mu
	  for(j=0;j<n_eigval;++j)
	    gsl_vector_set(x,j,-0.5/poolsize[i]/(gsl_vector_get(eig_val,j))*gsl_vector_get(yy,j));//x=-0.5/poolsize*sigma^-1*yy
	  gsl_blas_ddot (yy, x, &val);// val=yy^T x
	  logl+=val-0.5*(n_eigval*log(poolsize[i])+logdet);

	  for(h=0;h<n_haps;++h)
	    {
	      gsl_vector_memcpy (y_minus_mu,A[i]); //y_minus_mu=A[i]
	      gsl_blas_daxpy (-1.0, H[h],y_minus_mu); //y_minus_mu=A[i]-H[j]
	      gsl_blas_daxpy(-poolsize[i]+1,mu,y_minus_mu); //y_minus_mu=y_minus_mu-(poolsize-1)*mu
	      gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0,yy);//yy=u y_minus_mu

	      gsl_blas_dgemv (CblasTrans, 1.0, u,yy,0.0,y);//y=u^t yy
	      gsl_blas_daxpy (-1.0, y_minus_mu, y);
	      val2=gsl_blas_dnrm2(y);
	      if(val2>0.001)
		{not=1;}//not is 1 if A[i]-mu is not in the space spanned by columns of u
	      
	      //gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0, yy); //yy=1.0*u*y_minus_mu
	      for(j=0;j<n_eigval;++j)
		gsl_vector_set(x,j,-0.5/(poolsize[i]-1)/(gsl_vector_get(eig_val,j))*gsl_vector_get(yy,j));//x=-0.5/(poolsize-1)*sigma^-1*yy
	      gsl_blas_ddot (yy, x, &val2);// val2=yy^T x

	      v=poolsize[i]*p[h]*pow(1-1.0/(poolsize[i]),-0.5*n_eigval)*exp(val2-val); //coeff 2 has been dropped 
	      expected[h]+=v; 
	      sum+=v;
	    }
	}

      logl+=(-n_eigval*n_pools/2.0)*log(2*3.14159265); 
      fprintf(monitor,"%g\n",logl);
       for(j=0;j<n_haps;++j)
	 {
	   p[j]=expected[j]/sum;
	   //if(p[j]<1e-4/n_haps) p[j]=0.0;
	 }

       //if(fabs(sum-1.0)>0.0001)
       //{fprintf(stderr,"ERROR with sum of p %g != 1, iter %d\n",sum,iter);exit(1);}

      gsl_vector_free(yy);
      gsl_vector_free(x);
      gsl_matrix_free(u);

      //if(iter%100==99)
      printf("iter:%d, diff=%g\n",iter+1,logl-logl2);
      //if(logl-logl2<0.0){fprintf(stderr,"ERROR lkhood does not increase %g<%g !\n",logl,logl2);exit(1);}
      if(fabs(logl-logl2)<par.tol) go_on=0;
      //if(fabs(logl-logl2)<par.tol || logl-logl2<0.0) go_on=0;
      
      ++iter;
     
    }
  
  if(iter<par.n_iterations)
    printf("Exits after %d iterations, loglkhood difference %g\n",iter,logl-logl2);
  else
    printf("Exits since %d iterations were run, loglkhood difference=%g \n",iter,logl-logl2);
    
  gsl_vector_free(y);
  gsl_vector_free(y_minus_mu);

  results=fopen("AEML.out","w");    
  for(i=0;i<n_haps;++i)
    {
      //if(p[i]>1e-3)
	//{
	  for(j=0;j<n_loci;++j)
	    fprintf(results,"%g",gsl_vector_get(H[i],j));
	  fprintf(results," %8.6g\n",p[i]);
	//}
    }
  fclose(results);
  fclose(monitor);

  if(par.random_init == 1)
    { 
      monitor=fopen("AEML_seed","w");
      fprintf(monitor,"%ld\n",gsl_rng_get(rng));
      fclose(monitor);
    }

}


void read_parameters(FILE *in, struct PARAMETERS_STRUCT *par)
{
  //MP 280609

  char tag[40],target[40];
  char n_loci_found=0,n_pools_found=0,n_iterations_found=0;
  char tol_found=0,stab_found=0,random_init_found=0;
  char data_file_found=0,hap_file_found=0,found;

  while(!feof(in))
    {
      found=0;
      if(fscanf(in,"%s",tag)!=1)
	{if(feof(in)) break; fprintf(stderr,"ERROR with parameters file\n");exit(2);}

      sprintf(target,"n_loci");
      if(strcmp(tag,target)==0)
	{
	  if(n_loci_found)
	    {fprintf(stderr,"ERROR with parameters file: n_loci was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->n_loci)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_loci is invalid\n");exit(2);};
	  if(par->n_loci<2)
	    {fprintf(stderr,"ERROR with parameters file:value for n_loci is %d < 2\nExits!\n",par->n_loci);exit(2);}
	  n_loci_found=1;found=1;
	}

      sprintf(target,"n_pools");
      if(strcmp(tag,target)==0)
	{
	  if(n_pools_found)
	    {fprintf(stderr,"ERROR with parameters file: n_pools was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->n_pools)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_pools is invalid\n");exit(2);};
	  if(par->n_pools<1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_loci is %d < 1\nExits!\n",par->n_pools);exit(2);}
	  n_pools_found=1;found=1;
	}

      sprintf(target,"n_iterations");
      if(strcmp(tag,target)==0)
	{
	  if(n_iterations_found)
	    {fprintf(stderr,"ERROR with parameters file: n_iterations was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->n_iterations)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_iterations is invalid\n");exit(2);};
	  if(par->n_iterations<1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_iterations is %d < 1\nExits!\n",par->n_iterations);exit(2);}
	  n_iterations_found=1;found=1;
	}

      sprintf(target,"tol");
      if(strcmp(tag,target)==0)
	{
	  if(tol_found)
	    {fprintf(stderr,"ERROR with parameters file: tol was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->tol)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for tol is invalid\n");exit(2);};
	  if(par->tol<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for tol is %g < 0.0\nExits!\n",par->tol);exit(2);}
	  tol_found=1;found=1;
	}

  sprintf(target,"stab");
      if(strcmp(tag,target)==0)
  {
    if(stab_found)
      {fprintf(stderr,"ERROR with parameters file: stab was found several times\n");exit(2);}
    if(fscanf(in," %lf ",&par->stab)!=1)
      {fprintf(stderr,"ERROR with parameters file:value for stab is invalid\n");exit(2);};
    if(par->stab<0.0)
      {fprintf(stderr,"ERROR with parameters file:value for stab is %g < 0.0\nExits!\n",par->stab);exit(2);}
    stab_found=1;found=1;
  }

      sprintf(target,"hap_file");
      if(strcmp(tag,target)==0)
	{
	  if(hap_file_found)
	    {fprintf(stderr,"ERROR with parameters file: hap_file was found several times\n");exit(2);}
	  if(fscanf(in," %s ",&par->hap_file)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for hap_file is invalid\n");exit(2);};
	  hap_file_found=1;found=1;
	  par->read_haps=1;
	}

      sprintf(target,"data_file");
      if(strcmp(tag,target)==0)
	{
	  if(data_file_found)
	    {fprintf(stderr,"ERROR with parameters file: data_file was found several times\n");exit(2);}
	  if(fscanf(in," %s ",&par->data_file)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for data_file is invalid\n");exit(2);};
	  data_file_found=1;found=1;
	}

      sprintf(target,"random_init");
      if(strcmp(tag,target)==0)
	{
	  if(random_init_found)
	    {fprintf(stderr,"ERROR with parameters file: random_init was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->random_init)!=1 || (par->random_init != 0 && par->random_init != 1))
	    {fprintf(stderr,"ERROR with parameters file:value for random_init is invalid\n");exit(2);};
	  random_init_found=1;found=1;
	}

      if(found==0)
	{fprintf(stderr,"\nERROR with parameters file: unexpected label '%s' found\n\n",tag);exit(2);}
    }

  if(!data_file_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'data_file'.\n");exit(2);}  
  if(!n_loci_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'n_loci'.\n");exit(2);}
  if(!n_pools_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'n_pools'.\n");exit(2);}
  if(!n_iterations_found){
    fprintf(stderr,"Did not find label 'n_iterations' from parameters file.\n Thus, using 100000 iterations.\n");
    par->n_iterations=100000;}
 if(!tol_found){
   fprintf(stderr,"Did not find label 'tol' from parameters file.\n Thus, using default value 0.00001.\n");
   par->tol=0.00001;}
 if(!stab_found){
   fprintf(stderr,"Did not find label 'stab' from parameters file.\n Thus, using default value 0.0.\n");
   par->stab=0.0;}
  if(!hap_file_found){
    fprintf(stderr,"Did not find label 'hap_file' from parameters file.\n Thus, the initial list contains all haplotypes.\n");
    par->read_haps=0;
  }
 if(!random_init_found){
    fprintf(stderr,"Did not find label 'random_init' from parameters file.\n Thus, the freqs are initialized as 1/n_haps.\n");
    par->random_init=0;
  }



#if ECHO_INPUT == 1
  printf("Using the following parameters:\n");
  printf("n_loci=%d\nn_pools=%d\nn_iterations=%d\n",par->n_loci,par->n_pools,par->n_iterations);
  printf("tol=%g\nrandom_init=%d\n",par->tol,par->random_init);
  printf("stab=%g\n",par->stab);
  printf("data_file=%s\n",par->data_file);
  if(par->read_haps==1)
    printf("hap_file=%s\n",par->hap_file);
  else
    printf("using all haplotypes\n");
  printf("\n");
#endif

}




gsl_matrix *collect_eigen_directions(gsl_vector* eig_val,gsl_matrix *eig_vec,int n_loci,int *n_eigval)
{

  // At end eig_val[0,...,nonzero-1] contain the nonzero eigenvalues that correspond to
  // eigenvectors that are rows of u (nonzero x n_loci matrix)

  int nonzero=0,i,j,current=0;
  double tol=1e-6;
  gsl_matrix *u;

  for(i=0;i<n_loci;++i)
    if(gsl_vector_get(eig_val,i)>tol)
      ++nonzero;

  *n_eigval=nonzero;
  if(nonzero==0)
    {
      fprintf(stderr,"Covariance matrix is zero matrix!\n");
      return NULL;
    }

  u=gsl_matrix_alloc(nonzero,n_loci);
  for(i=0;i<n_loci;++i)
    if(gsl_vector_get(eig_val,i)>tol)
      {
	for(j=0;j<n_loci;++j)
	  gsl_matrix_set(u,current,j,gsl_matrix_get(eig_vec,j,i));
	gsl_vector_set(eig_val,current,gsl_vector_get(eig_val,i));
	++current;
      }

  return u; 
}

void opening_error(char *str)
{
  fprintf(stderr,"Can't open file: %s\nExits!\n",str);
  exit(5);
}

void alloc_error(char *str)
{
  fprintf(stderr,"Can't allocate memory: %s\nExits!\n",str);
  exit(2);
}

gsl_vector** all_haps(int *n_haps,int n_loci)
{
  int i,j,l;
  gsl_vector **H;

  *n_haps=1<<n_loci;

  if((H=(gsl_vector**) malloc((*n_haps)*sizeof(gsl_vector*)))==NULL)
    alloc_error("H");

  H[0]=gsl_vector_calloc(n_loci);
  for(i=1;i<*n_haps;++i)
    {
      H[i]=gsl_vector_alloc(n_loci);
      gsl_vector_memcpy (H[i],H[i-1]);
      l=0;
      while(gsl_vector_get(H[i],l)==1 && l<n_loci) ++l;
      if(l==n_loci) {fprintf(stderr,"ERROR with all_haps\n");exit(2);}
      gsl_vector_set(H[i],l--,1);
      while(l>=0)
	gsl_vector_set(H[i],l--,0);
    }

#if ECHO_INPUT == 1
  printf("\n\nUsing all haps on %d loci:\n",n_loci); 
  for(i=0;i<*n_haps;++i)
    {
      printf("hap %d: ",i+1);
      for(j=0;j<n_loci;++j)
	{
	  printf("%g",gsl_vector_get(H[i],j));
	}
      printf("\n");
    }

#endif

  return H;

}


gsl_vector** read_haps(FILE* in,int *n_haps,int n_loci)
{
  int i,j,a;
  gsl_vector** H;

  if(fscanf(in,"%d ",n_haps)!=1)
    {
      fprintf(stderr,"ERROR readind haplotype file!\nExits!\n");
      exit(5);
    }

  if(*n_haps<1)
    {
      fprintf(stderr,"ERROR negative number of haplotypes!\nExits!\n",*n_haps);
      exit(5);
    }

  if((H=(gsl_vector**) malloc((*n_haps)*sizeof(gsl_vector*)))==NULL)
    alloc_error("H");

  for(i=0;i<*n_haps;++i)
    {
      H[i]=gsl_vector_alloc(n_loci);
      for(j=0;j<n_loci;++j)
	{
	  if(fscanf(in,"%d",&a)!=1)
	    {
	      fprintf(stderr,"ERROR with hap file\nhaplotype %d locus %d\n",i+1,j+1);
	      exit(5);
	    }
	  if(a!=0 && a!=1)
	    {
	      fprintf(stderr,"ERROR with hap file\nhaplotype %d locus %d allele %d\n",i+1,j+1,a);
	      exit(5);
	    }
	  gsl_vector_set(H[i],j,a);
	}
    }

#if ECHO_INPUT == 1
  printf("\n\nRead the following haplotypes from file\n"); 
  for(i=0;i<*n_haps;++i)
    {
      printf("hap %d: ",i+1);
      for(j=0;j<n_loci;++j)
	{
	  printf("%g",gsl_vector_get(H[i],j));
	}
      printf("\n");
    }

#endif

  return H;
}

int** read_data(FILE *in,int n_loci,int n_pools)
{
  int **data;
  int i,j;

  /*  if(fscanf(in,"%d %d",n_loci,n_pools)!=2)
    {
      fprintf(stderr,"ERROR readind parameters file!\nExits!\n");
      exit(5);
    }
  */
  if(n_loci<2 || n_pools<1)
    {
      fprintf(stderr,"ERROR with negative input\nn_loci=%d n_pools=%d\n",n_loci,n_pools);
      exit(2);
    }
  
  if((data= (int**) malloc((n_pools)*sizeof(int*)))==NULL)
    alloc_error("data[]");

  for(i=0;i<n_pools;++i)
    if((data[i]=(int*) malloc((n_loci+1)*sizeof(int)))==NULL)
      alloc_error("data[][]");

  for(i=0;i<n_pools;++i)
    for(j=0;j<n_loci+1;++j)
      if(fscanf(in,"%d ",data[i]+j)!=1) 
	{fprintf(stderr,"READING ERROR at pool %d, locus %d\n",i+1,j);exit(2);}
  for(i=0;i<n_pools;++i)
    {
      if(data[i][0]<1)
	{
	  fprintf(stderr,"ERROR with poolsize (%d) at pool %d\n",data[i][0],i+1);
	  exit(1);
	}
      for(j=1;j<n_loci+1;++j)
	{	  
	  if(data[i][j]<0 || data[i][j]>2*data[i][0])
	    {
	      fprintf(stderr,"ERROR with allele count of pool %d at locus %d\n",i+1,j);
	      fprintf(stderr,"%d (poolsize =%d)\n\n",data[i][j],data[i][0]);
	      exit(1);
	    }
	}
   }

#if ECHO_INPUT == 1
  printf("Read the following data for %d pools and %d loci\n",n_pools,n_loci);
  for(i=0;i<n_pools;++i)
    {
      printf("Pool %d; size=%d\n",i+1,data[i][0]);
      for(j=1;j<n_loci+1;++j)
	printf("%d ",data[i][j]);
      printf("\n");
    }
#endif
 



  return data;

}

void check_rank_of_list(gsl_vector **H,int n_list,int n_loci,gsl_vector **A,int n_pools,double tol)
{
  //check and print out the rank of matrix H^t
  //if it is of full column rank then the problem has unique solution


  gsl_matrix *U,*V,*R;
  gsl_vector *s,*work,*y,*yy;
  int n,m,i,j,rank;
  double val;

  if(n_loci<n_list)
    {n=n_loci;m=n_list;}
  else
    {n=n_list;m=n_loci;}

  U=gsl_matrix_alloc(m,n);
  s=gsl_vector_alloc(n);
  V=gsl_matrix_alloc(n,n);
  work=gsl_vector_alloc(n);

  if(n==n_loci){
    for(i=0;i<m;++i)
      for(j=0;j<n;++j)
	gsl_matrix_set(U,i,j,gsl_vector_get(H[i],j));}//U=H
  else{
    for(i=0;i<m;++i)
      for(j=0;j<n;++j)
	gsl_matrix_set(U,i,j,gsl_vector_get(H[j],i));}//U=H^t

  gsl_linalg_SV_decomp (U,V,s,work); //requires that U is mxn where m>=n

  rank=0;
  for(i=0;i<n;++i)
    if(gsl_vector_get(s,i)>tol) ++rank;

  fprintf(stdout,"\n\nList has rank=%d, n_loci=%d, n_list=%d\n\n",rank, n_loci, n_list);

  R=gsl_matrix_alloc(n_loci,rank);
  if(n==n_loci)//span(H^t) is spanned by cols of V that correspond to the positive vals of s 
    {
      for(i=0;i<n_loci;++i)
	for(j=0;j<rank;++j)
	  gsl_matrix_set(R,i,j,gsl_matrix_get(V,i,j));
    }
  else
    {//span(H^t) is spanned by cols of U that correspond to the positive vals of s 
      for(i=0;i<n_loci;++i)
	for(j=0;j<rank;++j)
	  gsl_matrix_set(R,i,j,gsl_matrix_get(U,i,j));
    }

  y=gsl_vector_alloc(n_loci);
  yy=gsl_vector_alloc(rank);

  for(i=0;i<n_pools;++i)
    {
      gsl_vector_memcpy (y,A[i]); //y=A[i]
      gsl_blas_dgemv (CblasTrans, 1.0, R, y, 0.0,yy);//yy=R^t y
      gsl_blas_dgemv (CblasNoTrans, 1.0, R,yy,0.0,y);//y=R yy
      gsl_blas_daxpy (-1.0, A[i], y);
      val=gsl_blas_dnrm2(y);
      if(val>0.001)
	fprintf(stdout,"Pool %d NOT in list space (distance=%g)\n",i+1,val);
      else
	{
	  fprintf(stdout,"Pool %d in list space.",i+1);
	  if(rank==n_list) fprintf(stdout," Unique solution to the linear system exists!"); 
	  fprintf(stdout,"\n");
	}
    }

  gsl_matrix_free(U);
  gsl_vector_free(s);
  gsl_vector_free(y);
  gsl_vector_free(yy);
  gsl_matrix_free(V);
  gsl_matrix_free(R);
  gsl_vector_free(work);

}
