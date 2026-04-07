import { SvgColor } from 'src/components/svg-color';

// ----------------------------------------------------------------------

const icon = (name: string) => <SvgColor src={`/assets/icons/navbar/${name}.svg`} />;

export type NavItem = {
  title: string;
  path: string;
  icon: React.ReactNode;
  info?: React.ReactNode;
};

export const navData = [
  {
    title: 'Dashboard',
    path: '/',
    icon: icon('ic-analytics'),
  },
  {
    title: 'Monitoring',
    path: '/monitor',
    icon: icon('ic-lock'),
  },
  {
    title: 'Agentic actions',
    path: '/agentic',
    icon: icon('ic-blog'),
  },
  {
    title: 'ML & RAG',
    path: '/ml',
    icon: icon('ic-cart'),
  },
  {
    title: 'User',
    path: '/user',
    icon: icon('ic-user'),
  },
];
