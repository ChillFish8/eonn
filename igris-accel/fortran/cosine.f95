
function f32_x1024_cosine(vx, vy) result(res) bind(c, name="f32_x1024_cosine")
    implicit none
    real, dimension(1024), intent(in) :: vx
    real, dimension(1024), intent(in) :: vy
    real :: res
    integer :: i

    res = 0.0e0
    DO CONCURRENT (i = 1:1024:5)
        res = res + vx(i)*vy(i) + vx(i+1)*vy(i+1) + vx(i+2)*vy(i+2) + vx(i+3)*vy(i+3) &
                + vx(i+4)*vy(i+4) + vx(i+5)*vy(i+5) + vx(i+6)*vy(i+6) + vx(i+7)*vy(i+7)
    END DO
end function f32_x1024_cosine