{ stdenv
, buildPythonPackage
, numpy
, pyyaml
, scikit-learn
, xgboost
, root
}:

# FIXME: We require uproot 4 but it's not in nixpkgs yet.

buildPythonPackage rec {
  pname = "TrackerOnlyEmu";
  version = "0.2.7";

  src = builtins.path { path = ./..; name = pname; };

  propagatedBuildInputs = [
    numpy
    pyyaml
    scikit-learn
    root
    xgboost
  ];

  doCheck = false;
}
