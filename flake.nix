{
  description = "A collection of tools for emulating tracker responses.";

  inputs = {
    root-curated.url = "github:umd-lhcb/root-curated";
    nixpkgs.follows = "root-curated/nixpkgs";
    flake-utils.follows = "root-curated/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, root-curated }:
    {
      overlay = import ./nix/overlay.nix;
    }
    //
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ root-curated.overlay self.overlay ];
        };
        python = pkgs.python3;
        pythonPackages = python.pkgs;
      in
      {
        packages = flake-utils.lib.flattenTree {
          TrackerOnlyEmu = python.withPackages (p: with p; [ TrackerOnlyEmu ]);
        };
        devShell = pkgs.mkShell rec {
          name = "TrackerOnlyEmu-dev";
          buildInputs = with pythonPackages; [
            pkgs.clang-tools # For clang-format
            pkgs.root

            # Auto completion
            jedi

            # Linters
            flake8
            pylint

            # Python requirements (enough to get a virtualenv going).
            virtualenvwrapper

            # Pinned Python dependencies
            numpy
            xgboost
            scikit-learn
          ];

          shellHook = ''
            # Allow the use of wheels.
            SOURCE_DATE_EPOCH=$(date +%s)

            if test -d $HOME/build/python-venv; then
              VENV=$HOME/build/python-venv/${name}
            else
              VENV=./.virtualenv
            fi

            if test ! -d $VENV; then
              virtualenv $VENV
            fi
            source $VENV/bin/activate

            # allow for the environment to pick up packages installed with virtualenv
            export PYTHONPATH="$PWD:$VENV/${python.sitePackages}/:$PYTHONPATH"
          '';
        };
      });
}
