void print_final_results(FILE *out, FILE *in, int n_loci, double tol)
{
  int i,nhaps=0;
  char *hap=NULL;
  char **hlist=NULL;
  double *p=NULL,*sq=NULL,x,y,sum_p=0;

  if((hap=(char *) malloc((n_loci+1)*sizeof(char)))==NULL)
    alloc_error("print_final_results");
  
  fscanf(in,"%s",hap);

  while(!feof(in))
    {
      fscanf(in,"%lf %lf",&x,&y);
      sum_p+=x;
      for(i=0;i<nhaps;++i){if(!strcmp(hap,hlist[i])) break;}
      if(i<nhaps)
	{p[i]+=x;sq[i]+=y;}
      else
	{
	  ++nhaps;
	  if((hlist=(char **) realloc(hlist,nhaps*sizeof(char*)))==NULL)
	    alloc_error("print_final_results");
	  if((p=(double *) realloc(p,nhaps*sizeof(double)))==NULL)
	    alloc_error("print_final_results");
	  if((sq=(double *) realloc(sq,nhaps*sizeof(double)))==NULL)
	    alloc_error("print_final_results");
	  hlist[nhaps-1]=hap;
	  p[nhaps-1]=x;
	  sq[nhaps-1]=y;
	  if((hap=(char *) malloc((n_loci+1)*sizeof(char)))==NULL)
	    alloc_error("print_final_results");
	}
      fscanf(in,"%s",hap);
    }

  for(i=0;i<nhaps;++i)
    {
      if(p[i]/sum_p>tol)
	fprintf(out,"%s %8.6f %8.6f\n",hlist[i],p[i]/sum_p,sqrt(sq[i]/sum_p-p[i]*p[i]/sum_p/sum_p));
      free(hlist[i]);
    }

  free(hap);free(p);free(sq);free(hlist);

}


void read_parameters(FILE *in, struct PARAMETERS_STRUCT *par)
{
  //MP 240609

  char tag[40],target[40];
  char n_loci_found=0,n_pools_found=0,n_iterations_found=0,n_burnin_found=0,variable_list_found=0,write_trace_found=0,thin_found=0;
  char beta_a_found=0,tol_found=0,stab_found=0,beta_c_found=0,beta_mut_1_found=0,beta_mut_2_found=0,alpha_found=0,gamma_found=0,proba_mut_found=0;
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

      sprintf(target,"n_burnin");
      if(strcmp(tag,target)==0)
	{
	  if(n_burnin_found)
	    {fprintf(stderr,"ERROR with parameters file: n_burnin was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->n_burnin)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_burnin is invalid\n");exit(2);};
	  if(par->n_burnin<1)
	    {fprintf(stderr,"ERROR with parameters file:value for n_burnin is %d < 1\nExits!\n",par->n_burnin);exit(2);}
	  n_burnin_found=1;found=1;
	}

      sprintf(target,"variable_list");
      if(strcmp(tag,target)==0)
	{
	  if(variable_list_found)
	    {fprintf(stderr,"ERROR with parameters file:variable_list was found several times\n");exit(2);}
	  if(fscanf(in," %d ",&par->variable_list)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for variable_list is invalid\n");exit(2);};
	  if(par->variable_list>2 || par->variable_list<0)
	    {fprintf(stderr,"ERROR with parameters file:value for variable_list is %d != 0,1,2\nExits!\n",par->variable_list);exit(2);}
	  variable_list_found=1;found=1;
	}

      sprintf(target,"write_trace");
      if(strcmp(tag,target)==0)
  {
    if(write_trace_found)
      {fprintf(stderr,"ERROR with parameters file:write_trace was found several times\n");exit(2);}
    if(fscanf(in," %d ",&par->write_trace)!=1)
      {fprintf(stderr,"ERROR with parameters file:value for write_trace is invalid\n");exit(2);};
    if(par->write_trace>1 || par->write_trace<0)
      {fprintf(stderr,"ERROR with parameters file:value for write_trace is %d != 0,1\nExits!\n",par->write_trace);exit(2);}
    write_trace_found=1;found=1;
  }

  sprintf(target,"thin");
      if(strcmp(tag,target)==0)
  {
    if(thin_found)
      {fprintf(stderr,"ERROR with parameters file:thin was found several times\n");exit(2);}
    if(fscanf(in," %d ",&par->thin)!=1)
      {fprintf(stderr,"ERROR with parameters file:value for thin is invalid\n");exit(2);};
    if(par->thin<1)
      {fprintf(stderr,"ERROR with parameters file:value for thin is %d < 1\nExits!\n",par->thin);exit(2);}
    thin_found=1;found=1;
  }

      sprintf(target,"d");
      if(strcmp(tag,target)==0)
	{
	  if(beta_a_found)
	    {fprintf(stderr,"ERROR with parameters file: d was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->beta_a)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for d is invalid\n");exit(2);};
	  if(par->beta_a<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for d is %g < 0.0\nExits!\n",par->beta_a);exit(2);}
	  beta_a_found=1;found=1;
	}

      sprintf(target,"c");
      if(strcmp(tag,target)==0)
	{
	  if(beta_c_found)
	    {fprintf(stderr,"ERROR with parameters file: c was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->beta_c)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for c is invalid\n");exit(2);};
	  if(par->beta_c<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for c is %g < 0.0\nExits!\n",par->beta_c);exit(2);}
	  beta_c_found=1;found=1;
	}
      
      sprintf(target,"c_old");
      if(strcmp(tag,target)==0)
	{
	  if(beta_mut_1_found)
	    {fprintf(stderr,"ERROR with parameters file: c_old was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->beta_mut_1)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for c_old is invalid\n");exit(2);};
	  if(par->beta_mut_1<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for c_old is %g < 0.0\nExits!\n",par->beta_mut_1);exit(2);}
	  beta_mut_1_found=1;found=1;
	}

      sprintf(target,"c_new");
      if(strcmp(tag,target)==0)
	{
	  if(beta_mut_2_found)
	    {fprintf(stderr,"ERROR with parameters file: c_new was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->beta_mut_2)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for c_new is invalid\n");exit(2);};
	  if(par->beta_mut_2<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for c_new is %g < 0.0\nExits!\n",par->beta_mut_2);exit(2);}
	  beta_mut_2_found=1;found=1;
	}

      sprintf(target,"alpha");
      if(strcmp(tag,target)==0)
	{
	  if(alpha_found)
	    {fprintf(stderr,"ERROR with parameters file: alpha was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->alpha)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for alpha is invalid\n");exit(2);};
	  if(par->alpha<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for alpha is %g < 0.0\nExits!\n",par->alpha);exit(2);}
	  alpha_found=1;found=1;
	}

      sprintf(target,"gamma");
      if(strcmp(tag,target)==0)
	{
	  if(gamma_found)
	    {fprintf(stderr,"ERROR with parameters file: gamma was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->gamma)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for gamma is invalid\n");exit(2);};
	  if(par->gamma<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for gamma is %g < 0.0\nExits!\n",par->gamma);exit(2);}
	  gamma_found=1;found=1;
	}

      sprintf(target,"p_add");
      if(strcmp(tag,target)==0)
	{
	  if(proba_mut_found)
	    {fprintf(stderr,"ERROR with parameters file: p_add was found several times\n");exit(2);}
	  if(fscanf(in," %lf ",&par->proba_mut)!=1)
	    {fprintf(stderr,"ERROR with parameters file:value for p_add is invalid\n");exit(2);};
	  if(par->proba_mut<0.0)
	    {fprintf(stderr,"ERROR with parameters file:value for p_add is %g < 0.0\nExits!\n",par->proba_mut);exit(2);}
	  proba_mut_found=1;found=1;
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
      if(found==0)
	{fprintf(stderr,"\nERROR with parameters file: unexpected label '%s' found\n\n",tag);exit(2);}
    }

  if(!data_file_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'data_file'.\n");exit(2);}  
  if(!n_loci_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'n_loci'.\n");exit(2);}
  if(!n_pools_found){fprintf(stderr,"ERROR with parameters file: Did not find label 'n_pools'.\n");exit(2);}
  if(!n_iterations_found){
    fprintf(stderr,"Did not find label 'n_iterations' from parameters file.\n Thus, using 100000 iterations.\n");
    par->n_iterations=100000;}
  if(!n_burnin_found){
    fprintf(stderr,"Did not find label 'n_burnin' from parameters file.\n Thus, no burnin period in use.\n");
    par->n_burnin=0;}
  if(!variable_list_found){
    fprintf(stderr,"Did not find label 'variable_list' from parameters file.\n"); 
    fprintf(stderr,"Thus, additional haplotypes are allowed, but initial list is permanent.\n");
    par->variable_list=1;}
  if(!write_trace_found){
    fprintf(stderr,"Did not find label 'write_trace' from parameters file.\n"); 
    fprintf(stderr,"Thus, trace is not recorded.\n");
    par->write_trace=0;}
  if(!thin_found){
    fprintf(stderr,"Did not find label 'thin' from parameters file.\n"); 
    fprintf(stderr,"Thus, no thinning is performed.\n");
    par->thin=1;}
  if(!beta_a_found){
    fprintf(stderr,"Did not find label 'd' from parameters file.\n Thus, using default value 0.2.\n");
    par->beta_a=0.2;}
 if(!beta_c_found){
    fprintf(stderr,"Did not find label 'c' from parameters file.\n Thus, using default value 10.\n");
    par->beta_c=10;}
 if(!beta_mut_1_found){
    fprintf(stderr,"Did not find label 'c_old' from parameters file.\n Thus, using default value 10.\n");
    par->beta_mut_1=10;}
 if(!beta_mut_2_found){
    fprintf(stderr,"Did not find label 'c_new' from parameters file.\n Thus, using default value 2.\n");
    par->beta_mut_2=2;}
 if(!alpha_found){
    fprintf(stderr,"Did not find label 'alpha' from parameters file.\n Thus, using default value 0.00001.\n");
    par->alpha=0.00001;}
 if(!gamma_found){
    fprintf(stderr,"Did not find label 'gamma' from parameters file.\n Thus, using default value 8.\n");
    par->gamma=8;}
 if(!proba_mut_found){
    fprintf(stderr,"Did not find label 'p_add' from parameters file.\n Thus, using default value 0.5.\n");
    par->proba_mut=0.5;}
 if(!tol_found){
   fprintf(stderr,"Did not find label 'tol' from parameters file.\n Thus, using default value 0.001.\n");
   par->tol=0.001;}
  if(!stab_found){
   fprintf(stderr,"Did not find label 'stab' from parameters file.\n Thus, using default value 0.0.\n");
   par->stab=0.0;}
  if(!hap_file_found){
    fprintf(stderr,"Did not find label 'hap_file' from parameters file.\n Thus, the initial list contains all haplotypes.\n");
    par->read_haps=0;
  }

  if(par->n_burnin > par->n_iterations)
    {fprintf(stderr,"ERROR with parameters file: #burnin %d > #iterations %d\nExits!\n",par->n_burnin,par->n_iterations);exit(2);}

#if ECHO_INPUT == 1
  printf("Using the following parameters:\n");
  printf("n_loci=%d\nn_pools=%d\nn_iterations=%d\nn_burnin=%d\n",par->n_loci,par->n_pools,par->n_iterations,par->n_burnin);
  printf("alpha=%g\ngamma=%g\nc=%g\nd=%g\nc_old=%g\nc_new=%g\n",par->alpha,par->gamma,par->beta_c,par->beta_a,par->beta_mut_1,par->beta_mut_2);
  printf("p_add=%g\ntol=%g\n",par->proba_mut,par->tol);
  printf("stab=%g\n",par->stab);
  printf("variable_list=%d\n",par->variable_list);
  printf("write_trace=%d\n",par->write_trace);
  printf("thin=%d\n",par->thin);
  printf("data_file=%s\n",par->data_file);
  if(par->read_haps==1)
    printf("hap_file=%s\n",par->hap_file);
  else
    printf("using all haplotypes\n");
  printf("\n");

#endif
}



void sample_pair_from_proba(double *p,int n_haps,int *i1,int *i2,gsl_rng * rng)
{
  int i;
  double u,sum;

  u=gsl_rng_uniform(rng);
  sum=p[0];*i1=0;
  while(u>sum && *i1<n_haps) {++(*i1);sum+=p[*i1];}

  u=gsl_rng_uniform(rng)*(1-p[*i1]);
 
  if(*i1==0){*i2=1;sum=p[1];}
  else{*i2=0;sum=p[0];}
  while(u>sum && *i2<n_haps) {
    ++(*i2);
    if(*i2 == *i1) ++(*i2);
    sum+=p[*i2];
  }

  if(*i2<*i1) { i=*i1; *i1=*i2; *i2=i;}
  if(*i1<0 || *i1>=n_haps || *i2<0 || *i2>=n_haps || *i2<=*i1)
    {fprintf(stderr,"ERROR in sample_pair_from_proba");exit(2);}

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

void complete_list(gsl_vector **H,int *n_haps,int n_loci,gsl_vector **A,int n_pools,double tol)
{
  //checks and prints out the rank of matrix H^t
  //if it is of full column rank, then the problem has unique solution
  //if data is not in the space of H^t, then completes H by adding those
  //canonical base vectors that are not currently in space H^t

  //It would be better to check that A[i]-H^t*p belongs to span(sigma)
  //where p is initial frequencies (1/n_haps,1/n_haps,...,1/n_haps)
  //and sigma is the initial covariance matrix of loci.
  //This is not implemented here.

  gsl_matrix *U,*V,*R;
  gsl_vector *s,*work,*y,*yy,*e;
  int n,m,i,j,rank,data_not_in_space;
  double val;

  //arrange dimensions so that n <= m
  if(n_loci<*n_haps)
    {n=n_loci;m=*n_haps;}
  else
    {n=*n_haps;m=n_loci;}

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

  fprintf(stdout,"\n\nInitial list has rank=%d, n_loci=%d, n_list=%d\n\n",rank, n_loci, *n_haps);

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

  data_not_in_space=0;
  for(i=0;i<n_pools;++i)
    {
      gsl_vector_memcpy (y,A[i]); //y=A[i]
      gsl_blas_dgemv (CblasTrans, 1.0, R, y, 0.0,yy);//yy=R^t y
      gsl_blas_dgemv (CblasNoTrans, 1.0, R,yy,0.0,y);//y=R yy
      gsl_blas_daxpy (-1.0, A[i], y);
      val=gsl_blas_dnrm2(y);
      if(val>tol)
	{
	  fprintf(stdout,"Pool %d NOT in list space (distance=%g)\n",i+1,val);
	  data_not_in_space=1;
	}
      else
	{
	  fprintf(stdout,"Pool %d in list space.",i+1);
	  if(rank==*n_haps) fprintf(stdout," Unique solution to the linear system exists!"); 
	  fprintf(stdout,"\n");
	}
    }

  if(data_not_in_space)
    {
      e=gsl_vector_alloc(n_loci);
      for(i=0;i<n_loci;++i)
	{
	  gsl_vector_set_zero(y);
	  gsl_vector_set(y,i,1.0);
	  gsl_blas_dgemv (CblasTrans, 1.0, R, y, 0.0,yy);//yy=R^t y
	  gsl_blas_dgemv (CblasNoTrans, 1.0, R,yy,0.0,y);//y=R yy
	  gsl_vector_set_zero(e);
	  gsl_vector_set(e,i,1.0);
	  gsl_blas_daxpy (-1.0, e, y);
	  val=gsl_blas_dnrm2(y);
	  if(val>tol)
	    {
	      fprintf(stdout,"Adding base vector %i to list.\n",i+1);
	      gsl_vector_set_zero(H[*n_haps]);
	      gsl_vector_set(H[*n_haps],i,1.0);
	      ++(*n_haps);
	    }
	}
      free(e);
    }

  gsl_matrix_free(U);
  gsl_vector_free(s);
  gsl_vector_free(y);
  gsl_vector_free(yy);
  gsl_matrix_free(V);
  gsl_matrix_free(R);
  gsl_vector_free(work);

}

int sample_parent(gsl_vector **H,double *p,int i1,int n_haps,int n_loci,double *sampling_proba,gsl_rng * rng)
{
  int i,l,diff,*possible,j=0;
  double u,sum=0.0;

  if((possible=(int*) malloc(n_haps*sizeof(int)))==NULL)
    alloc_error("sample_parent: possible");

  for(i=0;i<n_haps;++i)
    {
      diff=0;
      for(l=0;l<n_loci;++l)
	{
	  if(gsl_vector_get(H[i1],l)!=gsl_vector_get(H[i],l))
	    ++diff;
	  if(diff>1) break;
	}
      if(diff==1)
	{
	  possible[j++]=i;
	  sum+=p[i];
	}
    }

  if(j==0) 
    {*sampling_proba=0.0;return -1;}

  u=gsl_rng_uniform(rng);
  u*=sum;

  i=0;
  while(u>0.0)
    {
      u-=p[possible[i]];
      ++i;
    }
  i=possible[i-1];
  *sampling_proba=p[i]/sum;

  free(possible);

  return i;

}


double coalescing_probability(gsl_vector **H,double *p,int i1,int i2,int n_haps,int n_loci)
{
  int i,l,diff;
  double sum=0.0;

  for(i=0;i<n_haps;++i)
    {
      diff=0;
      for(l=0;l<n_loci;++l)
	{
	  if(gsl_vector_get(H[i1],l)!=gsl_vector_get(H[i],l))
	    ++diff;
	  if(diff>1) break;
	}
      if(diff==1)
	sum+=p[i];
      if(i==i2)
	if(diff!=1) 
	  {fprintf(stderr,"ERROR: in coalescing probability, d(h(i1),h(12))>1\n");exit(19);}
    }

  return p[i2]/sum;

}

void print_to_results(FILE *results,gsl_vector **H,int i1,double *psum,double *p2sum)
{
  int i;

  for(i=0;i<H[i1]->size;++i)
    fprintf(results,"%g",gsl_vector_get(H[i1],i));
  fprintf(results," %g %g\n",psum[i1],p2sum[i1]);

}

int search_mutated_hap_from_H(gsl_vector **H,int n_haps,int *SNP_sums,int i1,int l)
{

  int i,k;
  int n_loci=H[i1]->size;
  int correct_SNP_sum=SNP_sums[i1]+(1-2*gsl_vector_get(H[i1],l));

  //for(i=0;i<n_loci;++i)
  //printf("%g",gsl_vector_get(H[i1],i));
  //printf("\n%d %d %d\n",l,SNP_sums[i1],correct_SNP_sum);
  for(i=0;i<n_haps;++i)
    {
      if((i==i1) || SNP_sums[i]!=correct_SNP_sum) continue;

      k=0;
      while(k<n_loci && gsl_vector_get(H[i],k)==gsl_vector_get(H[i1],k)) ++k;

      if(k!=l) continue;

      ++k;
      while(k<n_loci && gsl_vector_get(H[i],k)==gsl_vector_get(H[i1],k)) ++k;
      
      if(k==n_loci)
	return i;
    }
  return -1;
}

double log_2ton_minus_k(double n,double k,double tol)
{
  //returns log(2^n-k),assuming k<2^n
  //uses: log(2^n-k)=log(2^n(1-k*2^(-n)))=n*log(2)+log(1-k*2^(-n))
  //=n*log(2)-(k*2^(-n))-1/2*(k*2^(-n))^2-1/3*(k*2^(-n))^3-1/4*(k*2^(-n))^4-...
  //stops when terms are less than tol

  double res,v,u;
  int i;

  res=n*log(2);
  v=log(k)-n*log(2);
  if(v>0) {fprintf(stderr,"ERROR in log_2ton_minus_k: k<2^n does not hold\nExits!\n");exit(8);}
  u=v;
  i=1;
  while(exp(u)>tol)
    {
      res-=1.0/i*exp(u);
      ++i;
      u=i*v;
    }
  return res;

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


double loglkhood(gsl_vector **A,gsl_vector * mu,gsl_matrix* u,gsl_vector *eig_val,int *poolsize,int n_pools,int n_loci,int n_eigval,int *not_in_span_u)
{
  //returns loglikelihood of pooled data,
  //we would not need to calculate all constants, since they cancel in HR
  //but they are here for completeness
  double logl=0.0,val,logdet;
  gsl_vector *y,*x,*yy,*y_minus_mu;
  int i,j,not=0;

  y=gsl_vector_alloc(n_loci);
  y_minus_mu=gsl_vector_alloc(n_loci);
  yy=gsl_vector_alloc(n_eigval);
  x=gsl_vector_alloc(n_eigval);

  logdet=0.0;
  for(i=0;i<n_eigval;++i)
    logdet+=log(gsl_vector_get(eig_val,i));


  for(i=0;i<n_pools;++i)
    {
      gsl_vector_memcpy (y_minus_mu,A[i]); //y_minus_mu=A[i]
      gsl_blas_daxpy(-poolsize[i],mu,y_minus_mu); //y=y_minus_mu-poolsize*mu
      gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0,yy);//yy=u y
      gsl_blas_dgemv (CblasTrans, 1.0, u,yy,0.0,y);//y=u^t yy
      gsl_blas_daxpy (-1.0, y_minus_mu, y);
      val=gsl_blas_dnrm2(y);
      //if(val<0.001){printf("OK:%d %g\n",i+1,val);}else
      if(val>0.001)
	{not=1;}//not is 1 if A[i]-mu is not in the space spanned by columns of u

      //gsl_blas_dgemv (CblasNoTrans, 1.0, u, y_minus_mu, 0.0, yy); //yy=1.0*u*y_minus_mu
      for(j=0;j<n_eigval;++j)
	gsl_vector_set(x,j,-0.5/poolsize[i]/(gsl_vector_get(eig_val,j))*gsl_vector_get(yy,j));//x=-0.5/poolsize*sigma^-1*yy
      gsl_blas_ddot (yy, x, &val);// val=y^T x
      logl+=val-0.5*(n_eigval*log(poolsize[i])+logdet); //n_loci*log(poolsize) could be omitted
   }
  *not_in_span_u=not;

  gsl_vector_free(y);
  gsl_vector_free(y_minus_mu);
  gsl_vector_free(yy);
  gsl_vector_free(x);

  //this constant can be ignored for the purpose of the algorithm
  logl+=(-n_eigval*n_pools/2.0)*log(2*3.14159265); 
  return logl;
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

