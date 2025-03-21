for d in $(find . -maxdepth 2 -type d)
do
  pushd $d
  rm -rf a.* a a/ *-interface.* *.out exec_time.txt *.png *.o *.isa *_genx.cpp signed* temp* profile.mon
  popd
done
