/** Only the two strict assertion operations used by the standing controller. */
function fail(message){throw new Error(message??'Recovery invariant failed');}
export default Object.freeze({
  equal(actual,expected,message){if(!Object.is(actual,expected))fail(message);},
  ok(value,message){if(!value)fail(message);}
});
