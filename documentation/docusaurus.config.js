// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const { themes } = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
    title: 'Gomoku AI',
    tagline: 'Advanced AI for the Gomoku Game',
    favicon: 'img/favicon.ico',

    url: 'https://yanisdolivet.github.io', // Your GitHub Pages base URL
    // For local development use '/' and for GitHub Pages use '/gomoku/'.
    // Make it configurable with the DOCS_BASE_URL env var so you can run
    // `DOCS_BASE_URL=/gomoku/ npm run build` when deploying.
    baseUrl: process.env.DOCS_BASE_URL || '/',

    // GitHub Pages deployment config
    organizationName: 'yanisdolivet', // Your GitHub username
    projectName: 'gomoku',       // The exact name of your repository
    deploymentBranch: 'gh-pages',
    trailingSlash: false,

    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'warn',

    i18n: {
        defaultLocale: 'en',
        locales: ['en'],
    },

    presets: [
        [
            'classic',
            /** @type {import('@docusaurus/preset-classic').Options} */
            ({
                docs: {
                    sidebarPath: require.resolve('./sidebars.js'),
                    // Edit this page link
                    editUrl:
                        'https://github.com/yanisdolivet/gomoku/tree/main/documentation/',
                },
                blog: false,
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
            }),
        ],
    ],

    themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            // Social image
            image: 'img/logo.png',
            announcementBar: {
                id: 'welcome',
                content:
                    '<strong>Welcome to the Gomoku AI Documentation.</strong>',
                backgroundColor: '#245d88',
                textColor: '#fff',
                isCloseable: true,
            },
            navbar: {
                title: 'Gomoku AI',
                logo: {
                    alt: 'Gomoku AI Logo',
                    src: 'img/logo.png',
                },
                items: [
                    // User Manual
                    {
                        type: 'docSidebar',
                        sidebarId: 'tutorialSidebar',
                        position: 'left',
                        label: 'Manual',
                    },
                    // Developer API (Doxygen)
                    {
                        to: 'pathname:///api/index.html',
                        label: 'API Reference',
                        position: 'left',
                    },
                    // GitHub
                    {
                        href: 'https://github.com/yanisdolivet/gomoku',
                        label: 'GitHub',
                        position: 'right',
                    },
                ],
            },
            footer: {
                style: 'dark',
                links: [
                    {
                        title: 'Documentation',
                        items: [
                            {
                                label: 'User Manual',
                                to: '/docs/Introduction',
                            },
                            {
                                label: 'API Reference',
                                href: 'pathname:///api/index.html',
                            },
                        ],
                    },
                    {
                        title: 'Community',
                        items: [
                            {
                                label: 'Epitech',
                                href: 'https://epitech.eu',
                            },
                            {
                                label: 'GitHub',
                                href: 'https://github.com/yanisdolivet/gomoku',
                            },
                        ],
                    },
                ],
                copyright: `Copyright © ${new Date().getFullYear()} Gomoku Project. Built with Docusaurus & C++.`,
            },
            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
                additionalLanguages: ['cpp', 'bash', 'makefile'],
            },
        }),
};

module.exports = config;