export interface NavItem {
  label: string;
  subLabel?: string;
  children?: Array<NavItem>;
  href?: string;
  openNewTab?: boolean;
}

export const NAV_ITEMS: Array<NavItem> = [
  {
    label: "Home",
    href: "/",
  },
  {
    label: "Explore",
    href: "/explore",
  },
  {
    label: "Github",
    children: [
      {
        label: "Frontend",
        subLabel: "degtrdg/bioconceptvec-explorer",
        href: "https://github.com/degtrdg/bioconceptvec-explorer",
        openNewTab: true,
      },
      {
        label: "Backend",
        subLabel: "degtrdg/bioconceptvec-explorer-frontend",
        href: "https://github.com/degtrdg/bioconceptvec-explorer-frontend",
        openNewTab: true,
      },
    ],
  },
];
