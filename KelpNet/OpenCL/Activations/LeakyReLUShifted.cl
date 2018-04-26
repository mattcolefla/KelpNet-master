Real ForwardActivate(Real gpuY)
{
	const Real a = 0.001;
    const Real offset = 0.5;

    Real y;
    if (gpuY + offset > 0.0)
    {
        y = gpuY + offset;
    }
    else
    {
        y = (gpuY + offset) * a;
    }
    return y;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpuY <= 0.0 ? gpuY * /*slope*/ + 0 : gpugX;
}