/** Status pill beside a player's points: "Final", or a pulsing dot plus the
 * game clock while live (both green). Nothing pre-kickoff -- that number is
 * still a projection, shown dimmer instead (see .roster-row__pts). */
export function LiveTag({ status, clock }) {
  if (status === "final") return <span className="live-pill">Final</span>;
  if (status === "in_progress") {
    return (
      <span className="live-pill">
        <span className="live-dot" aria-hidden="true" />
        {clock}
      </span>
    );
  }
  return null;
}
