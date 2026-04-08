import type { Theme, SxProps } from '@mui/material/styles';
import type { TimeSortOrder } from 'src/utils/table-time-sort';

import TableCell from '@mui/material/TableCell';
import TableSortLabel from '@mui/material/TableSortLabel';

import { toggleTimeSortOrder } from 'src/utils/table-time-sort';

type Props = {
  label: string;
  order: TimeSortOrder;
  onOrderChange: (next: TimeSortOrder) => void;
  align?: 'left' | 'right' | 'center';
  sx?: SxProps<Theme>;
};

export function TimeSortHeadCell({ label, order, onOrderChange, align = 'left', sx }: Props) {
  return (
    <TableCell align={align} sortDirection={order} sx={sx}>
      <TableSortLabel active direction={order} onClick={() => onOrderChange(toggleTimeSortOrder(order))}>
        {label}
      </TableSortLabel>
    </TableCell>
  );
}
