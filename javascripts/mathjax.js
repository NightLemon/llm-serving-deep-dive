window.MathJax = {
  tex: {
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();
      typesetMath();
    }
  }
};

let typesetPromise = Promise.resolve();

function typesetMath() {
  const mathJax = window.MathJax;
  if (typeof mathJax?.typesetPromise !== "function") {
    return;
  }

  typesetPromise = typesetPromise
    .then(() => {
      mathJax.startup?.output?.clearCache?.();
      mathJax.typesetClear?.();
      mathJax.texReset?.();
      return mathJax.typesetPromise();
    })
    .catch((error) => {
      console.error("MathJax typeset failed", error);
    });
}

document$.subscribe(() => {
  typesetMath();
});
