# Molecule: N2
cd n2/
cd dz/
# write integrals
echo "N2 cc-pVDZ"
#python -W ignore write_integrals.py 
# Do all hf, mp2, and ccsd(t) jobs, timing them.
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore hf_2.py
/usr/bin/time python -W ignore hf_3.py
/usr/bin/time python -W ignore hf_4.py
/usr/bin/time python -W ignore hf_5.py
/usr/bin/time python -W ignore hf_6.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore mp2_2.py
/usr/bin/time python -W ignore mp2_3.py
/usr/bin/time python -W ignore mp2_4.py
/usr/bin/time python -W ignore mp2_5.py
/usr/bin/time python -W ignore mp2_6.py
/usr/bin/time python -W ignore ccsdt_1.py
/usr/bin/time python -W ignore ccsdt_2.py
/usr/bin/time python -W ignore ccsdt_3.py
/usr/bin/time python -W ignore ccsdt_4.py
/usr/bin/time python -W ignore ccsdt_5.py
/usr/bin/time python -W ignore ccsdt_6.py
## Go back to TZ
cd ..
cd tz/
echo "N2 cc-pVTZ"
# write integrals
#python -W ignore write_integrals.py 
# Do all hf, mp2, and ccsd(t) jobs, timing them.
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore hf_2.py
/usr/bin/time python -W ignore hf_3.py
/usr/bin/time python -W ignore hf_4.py
/usr/bin/time python -W ignore hf_5.py
/usr/bin/time python -W ignore hf_6.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore mp2_2.py
/usr/bin/time python -W ignore mp2_3.py
/usr/bin/time python -W ignore mp2_4.py
/usr/bin/time python -W ignore mp2_5.py
/usr/bin/time python -W ignore mp2_6.py
/usr/bin/time python -W ignore ccsdt_1.py
/usr/bin/time python -W ignore ccsdt_2.py
/usr/bin/time python -W ignore ccsdt_3.py
/usr/bin/time python -W ignore ccsdt_4.py
/usr/bin/time python -W ignore ccsdt_5.py
/usr/bin/time python -W ignore ccsdt_6.py
cd ..
cd ..

