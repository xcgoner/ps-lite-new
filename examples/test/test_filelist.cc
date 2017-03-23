#include <iostream>
#include <dirent.h>
#include <cstring>

using namespace std;

void SplitFilename (const string& str, string& folder, string& prefix)
{
  size_t found;
  found = str.find_last_of("/\\");
  folder = str.substr(0, found);
  prefix = str.substr(found + 1);
}

int main(int argc, char *argv[]) {

  string fileprefix = "/home/cx2/ClionProjects/ps-lite-new/examples/LR_proximal/script/a9a-data/train/part-";
  string folder, prefix;
  SplitFilename(fileprefix, folder, prefix);
  cout << folder << endl;
  cout << prefix << endl;

  DIR *dpdf;
  struct dirent *epdf;
  dpdf = opendir(folder.c_str());
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
      if (strncmp(epdf->d_name, prefix.c_str(), prefix.length()) == 0) {
        cout << folder + "/" + string(epdf->d_name) << endl;
      }
    }
  }

  return 0;
}