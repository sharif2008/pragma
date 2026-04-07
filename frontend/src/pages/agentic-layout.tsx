import { Outlet } from 'react-router-dom';

/** Parent route for `/agentic` — list at index, full report at `report/:publicId`. */
export default function AgenticLayoutPage() {
  return <Outlet />;
}
