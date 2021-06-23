function val = fast_poisson(mu)

%% Basically just fish reorganised for speed.
%% Also fixed a bug in fish: see the rounding with the normal
%% approximation.
%% Now works with higher dimensional arrays too. (original fish
%% assumed 2D matrices.)
%%
%% 
%% Summary of bugs:
%% 
%% RGL's poisson.m:
%%   while somecondition:
%%     noise = noise + 1
%%     for a large mean, there's a final
%%       noise = noise + round(mean + gauss_rnd_var)
%% 
%% Shouldn't keep adding for a large mean, although it seems fine on a
%% very casual inspection
%% 
%% >> poisson([1.2 2.3 300.4])
%% ans =
%%      2     3   315
%% >> poisson([1.2 2.3 300.4])
%% ans =
%%      2     2   325
%% >> poisson([1.2 2.3 300.4])
%% ans =
%%      0     4   284
%% >> poisson([100 200 300.4])
%% ans =
%%    100   198   273
%% >> poisson([100 200 300.4])
%% ans =
%%    103   185   325
%% >> poisson([100 200 300.4 400 500])
%% ans =
%%     92   207   304   382   451
%%
%% But when there're small and large values in the same matrix,
%% 
%% >> x = poisson([49 ones(1,1000000)*300]);
%% >> x(1)
%% ans =
%%     35
%% >> mean(x(2:end))
%% ans =
%%   334.9800
%% 
%% The mean of these large values = 300 + 35.
%%
%% So in conclusion, poisson.m should have
%%    noise = round(mean + gauss_rnd_var)
%%
%% ========================================
%% 
%% fish.m:
%%   ...
%%   for a large mean:
%%     noise = mean + round(gauss_rnd_var)
%% 
%% This seems to assume that 'mean' is an integer value. fish.m is
%% correct for small means, but wrong for large means.
%%    >> fish([1.2 2.3 3.4])
%%    ans =
%%         2     2     6
%%    >> fish([1.2 2.3 300.4])
%%    ans =
%%        1.0000    2.0000  315.4000
%%   
%% Need to round the whole expression instead:
%%     round(mean + gauss_rnd_var)
%%    
%% ========================================

%% val = mu(:);
%% for k = 1:size(val,1)
%%     val(k) = poiss1(val(k));
%% end;
%% 
%% %% val = poiss1(mu(:));
%% 
%% val = reshape(val, size(mu));
%% 
%% function noise = poiss1(mu1)
%% 
%% if mu1 > 100
%%     noise=round(mu1+sqrt(mu1)*randn(1));
%% else
%%     value = exp(-mu1);
%%     noise = 0;
%%     current = rand(1);
%% 
%%     while(current > value)
%%          noise = noise + 1;
%%          current = current * rand(1);
%%     end
%% end
%% 

%% Damn, just can't beat vectorization
%% The Lisp version was much much slower in clisp (clisp-hs).
%%
%% SBCL also threw up a lot of optimization problems because the code
%% is not well done enough to allow for good type-checking. ie more
%% room for improvement.
%%
%% Profiling times:
%% 
%% The first one used up real time because it has been paged out to
%% disk and needed to be paged back in.
%% CL-USER> (time (loop repeat 1e6 do (poisson-random 5.0d0)))
%% Evaluation took:
%%   5.235 seconds of real time
%%   1.046875 seconds of user run time
%%   0.046875 seconds of system run time
%%   [Run times include 0.223 seconds GC run time.]
%%   0 calls to %EVAL
%%   0 page faults and
%%   208,193,408 bytes consed.
%% NIL
%% CL-USER> (time (loop repeat 1e6 do (poisson-random 5.0d0)))
%% Evaluation took:
%%   1.219 seconds of real time
%%   1.078125 seconds of user run time
%%   0.0 seconds of system run time
%%   [Run times include 0.218 seconds GC run time.]
%%   0 calls to %EVAL
%%   0 page faults and
%%   207,898,736 bytes consed.
%% NIL
%% CL-USER> (time (loop repeat 1e6 do (poisson-random 5.0d0)))
%% Evaluation took:
%%   1.203 seconds of real time
%%   1.03125 seconds of user run time
%%   0.015625 seconds of system run time
%%   [Run times include 0.139 seconds GC run time.]
%%   0 calls to %EVAL
%%   0 page faults and
%%   207,944,568 bytes consed.
%% NIL
%% CL-USER> 
%% 
%%(time (loop repeat (* 32 32 8 8) do (poisson-random 5.0d0)))
%% Evaluation took:
%%   0.078 seconds of real time
%%   0.078125 seconds of user run time
%%   0.0 seconds of system run time
%%   0 calls to %EVAL
%%   0 page faults and
%%   13,631,488 bytes consed.
%% NIL
%% CL-USER> (lisp-implementation-type)
%% "SBCL"
%% CL-USER> (lisp-implementation-version)
%% "1.0.6"
%% CL-USER> 
%%
%% Contrast this with clisp on top of cygwin.
%% 
%% CL-USER> (time (loop repeat 1e6 do (poisson-random 5.0d0)))
%% Real time: 85.457 sec.
%% Run time: 85.218 sec.
%% Space: 2048121544 Bytes
%% GC: 137, GC time: 39.583 sec.
%% NIL
%% CL-USER> (time (loop repeat 1e6 do (poisson-random 5.0d0)))
%% Real time: 85.386 sec.
%% Run time: 85.235 sec.
%% Space: 2048411788 Bytes
%% GC: 138, GC time: 39.982 sec.
%% NIL
%% CL-USER> 
%% 
%% 
%% CL-USER> (time (loop repeat (* 32 32 8 8) do (poisson-random 5.0d0)))
%% Real time: 5.207 sec.
%% Run time: 5.203 sec.
%% Space: 132645560 Bytes
%% GC: 8, GC time: 2.281 sec.
%% NIL
%% CL-USER> (lisp-implementation-type)
%% "CLISP"
%% CL-USER> (lisp-implementation-version)
%% "2.41 (2006-10-13) (built 3384478173) (memory 3384479010)"
%% CL-USER> 
%% 
%% Here's Matlab.
%%
%% >> tic;fast_poisson(ones(1e6,1)*5);toc
%% Elapsed time is 2.195988 seconds.
%% >> tic;fast_poisson(ones(1e6,1)*5);toc
%% Elapsed time is 2.190031 seconds.
%% >> tic;fast_poisson(ones(1e6,1)*5);toc
%% Elapsed time is 2.305031 seconds.
%%
%% Finally, we have:
%%
%% >> tic;fish(ones(1e6,1)*5);toc
%% Elapsed time is 4.016708 seconds.
%% >> tic;fish(ones(1e6,1)*5);toc
%% Elapsed time is 3.708337 seconds.
%% 
%%
%% Conclusion? We can go faster by not using Matlab, but it's probably
%% fast enough: it is 2x slower than the poisson in cl-variates
%% (SBCL), which is probably 2-5x slower (just a guess) than the ideal
%% fully optimised case.

value = exp(-mu);
noise = zeros(size(mu));

current = rand(size(mu));

hits = current > value;
while(any(hits(:)))
     noise = noise + hits;
     current = current .* rand(size(mu));
     hits = current > value;
end

bigmean = find(mu > 100);
mb = mu(bigmean);
noise(bigmean) = round(mb + sqrt(mb).*randn(size(bigmean)));

val = noise;
