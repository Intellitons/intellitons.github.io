(() => {
  const KEY = "intelliton-lang";

  const preferredLanguage = () => {
    const saved = window.localStorage.getItem(KEY);
    if (saved === "en" || saved === "zh") {
      return saved;
    }

    return navigator.language && navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
  };

  const applyLanguage = (lang) => {
    document.body.dataset.activeLang = lang;
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";

    document.querySelectorAll("[data-set-lang]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === lang));
    });

    const pageTitle = lang === "zh"
      ? document.body.dataset.pageTitleZh || document.body.dataset.pageTitleEn
      : document.body.dataset.pageTitleEn;

    if (pageTitle) {
      document.title = `${pageTitle} | Intelliton Blog`;
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    const initial = preferredLanguage();
    applyLanguage(initial);

    document.querySelectorAll("[data-set-lang]").forEach((button) => {
      button.addEventListener("click", () => {
        const lang = button.dataset.setLang;
        window.localStorage.setItem(KEY, lang);
        applyLanguage(lang);
      });
    });
  });
})();